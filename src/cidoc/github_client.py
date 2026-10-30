"""Thin httpx client for the GitHub REST API."""

from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr

API_VERSION = "2022-11-28"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class GitHubClient:
    def __init__(
        self,
        token: SecretStr,
        base_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # Log downloads answer 302 to a signed blob URL on another host;
        # httpx drops the Authorization header on cross-origin redirects.
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "ci-doctor",
            },
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"GitHubClient(base_url={str(self._http.base_url)!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        response = self._http.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", url, params=params).json()

    def download_log(self, url: str) -> bytes:
        return self.request("GET", url).content
