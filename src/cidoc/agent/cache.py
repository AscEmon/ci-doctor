"""On-disk cache of agent tool results.

A result depends only on what its tool reads, and that never changes:
- repository tools read at a fixed commit, keyed by (repo, head_sha, tool, args);
- run tools read one finished run, keyed by (repo, run_id, tool, args);
- log tools read the stored log, keyed by job id plus a hash of the log (a refetch with
  newer redaction rules replaces the stored log, and with it the key).
So entries never expire, and an eval re-run makes no GitHub call for a cached result.
Only successful results are cached: an error (404, rate limit) is retried next time.

Entries hold the redacted, uncapped text; the output cap is applied on every read.
One JSON file per key under <data_dir>/cache/agent_tools/ (data/ is never committed).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CACHE_DIRNAME = "cache/agent_tools"
FORMAT_VERSION = 1


def tool_cache_key(repo: str, ref: str, tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps([FORMAT_VERSION, repo, ref, tool, args], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

    @classmethod
    def in_data_dir(cls, data_dir: Path) -> "ToolCache":
        return cls(data_dir / CACHE_DIRNAME)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> str | None:
        """The cached text, or None (also for an unreadable entry)."""
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            text = entry["output"]
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None
        if not isinstance(text, str):
            self.misses += 1
            return None
        self.hits += 1
        return text

    def put(self, key: str, text: str, meta: dict[str, Any]) -> None:
        """Write atomically, so an interrupted run never leaves a half-written entry."""
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, **meta, "output": text}
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
