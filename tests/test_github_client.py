import httpx
import pytest
from pydantic import SecretStr

from cidoc.github_client import GitHubClient, RetryableHTTPError

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


def make_retrying_client(
    responses: list[httpx.Response], sleeps: list[float], max_attempts: int = 5
) -> tuple[GitHubClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[len(seen) - 1]

    client = GitHubClient(
        FAKE_TOKEN,
        transport=httpx.MockTransport(handler),
        max_attempts=max_attempts,
        sleep=sleeps.append,
    )
    return client, seen


def test_retries_5xx_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    responses = [httpx.Response(502), httpx.Response(503), httpx.Response(200, json={})]
    client, seen = make_retrying_client(responses, sleeps)

    with client:
        assert client.get_json("/x") == {}

    assert len(seen) == 3
    assert sleeps == [1.0, 2.0]


def test_secondary_rate_limit_honors_retry_after() -> None:
    sleeps: list[float] = []
    responses = [
        httpx.Response(403, headers={"Retry-After": "7"}, json={"message": "slow down"}),
        httpx.Response(200, json={}),
    ]
    client, _ = make_retrying_client(responses, sleeps)

    with client:
        client.get_json("/x")

    assert sleeps == [7.0]


def test_secondary_rate_limit_without_retry_after_waits_a_minute() -> None:
    sleeps: list[float] = []
    responses = [
        httpx.Response(429, json={"message": "You have exceeded a secondary rate limit."}),
        httpx.Response(200, json={}),
    ]
    client, _ = make_retrying_client(responses, sleeps)

    with client:
        client.get_json("/x")

    assert sleeps == [60.0]


def test_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    client, seen = make_retrying_client([httpx.Response(500)] * 3, sleeps, max_attempts=3)

    with client, pytest.raises(RetryableHTTPError):
        client.get_json("/x")

    assert len(seen) == 3


def test_does_not_retry_client_errors() -> None:
    sleeps: list[float] = []
    client, seen = make_retrying_client([httpx.Response(404)], sleeps)

    with client, pytest.raises(httpx.HTTPStatusError):
        client.get_json("/x")

    assert len(seen) == 1
    assert sleeps == []
