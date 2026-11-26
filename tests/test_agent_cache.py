"""Agent tool cache: results at a fixed commit, run or log are reused, never refetched."""

import json
from pathlib import Path
from typing import Any

from cidoc.agent.cache import ToolCache, tool_cache_key
from cidoc.agent.tools import TRUNCATED, ToolBox, ToolEnv
from tests.agent_fakes import SHA, FakeGitHub, contents, context, file_entry, http_error

PUBSPEC = contents("pubspec.yaml")


def toolbox(github: FakeGitHub, cache: ToolCache, **ctx: Any) -> ToolBox:
    return ToolBox(ToolEnv(context(**ctx), github), cache=cache)


def test_a_cached_result_makes_no_github_call(tmp_path: Path) -> None:
    first = FakeGitHub({PUBSPEC: file_entry("name: app\n")})
    out = toolbox(first, ToolCache(tmp_path)).call("read_file", {"path": "pubspec.yaml"})
    assert not out.cached

    again = FakeGitHub()  # would 404 on any request
    cache = ToolCache(tmp_path)
    hit = toolbox(again, cache).call("read_file", {"path": "./pubspec.yaml", "start_line": 1})

    assert again.calls == []
    assert hit.cached
    assert hit.ok
    assert hit.text == out.text
    assert cache.hits == 1


def test_arguments_are_normalized_before_keying(tmp_path: Path) -> None:
    github = FakeGitHub({PUBSPEC: file_entry("name: app\n")})
    tools = toolbox(github, ToolCache(tmp_path))

    tools.call("read_file", {"path": "pubspec.yaml"})
    tools.call("read_file", {"path": "pubspec.yaml", "start_line": 1, "end_line": 200})

    assert len(github.calls) == 1


def test_results_are_keyed_by_commit_and_run(tmp_path: Path) -> None:
    cache = ToolCache(tmp_path)
    github = FakeGitHub({PUBSPEC: file_entry("name: app\n")})
    toolbox(github, cache).call("read_file", {"path": "pubspec.yaml"})

    toolbox(github, cache, head_sha="b" * 40).call("read_file", {"path": "pubspec.yaml"})
    toolbox(github, cache, repo="o/other").call("read_file", {"path": "pubspec.yaml"})

    assert len(github.calls) == 3  # a new commit or repo is a new key
    assert tool_cache_key("o/a", f"commit:{SHA}", "x", {}) != tool_cache_key(
        "o/a", "run:70", "x", {}
    )


def test_errors_are_not_cached(tmp_path: Path) -> None:
    cache = ToolCache(tmp_path)
    failing = FakeGitHub({PUBSPEC: http_error(403, retryable=True)})
    assert not toolbox(failing, cache).call("read_file", {"path": "pubspec.yaml"}).ok

    working = FakeGitHub({PUBSPEC: file_entry("name: app\n")})
    out = toolbox(working, cache).call("read_file", {"path": "pubspec.yaml"})

    assert out.ok
    assert not out.cached
    assert len(working.calls) == 1


def test_log_results_change_when_the_stored_log_changes(tmp_path: Path) -> None:
    cache = ToolCache(tmp_path)
    args = {"start_line": 1, "end_line": 1}
    before = toolbox(FakeGitHub(), cache, log_lines=["old"]).call("read_log", args)

    after = toolbox(FakeGitHub(), cache, log_lines=["refetched"]).call("read_log", args)

    assert "L1| old" in before.text
    assert not after.cached
    assert "L1| refetched" in after.text


def test_the_cap_is_applied_on_read_not_stored(tmp_path: Path) -> None:
    text = "".join(f"{'z' * 70}\n" for _ in range(150))
    cache = ToolCache(tmp_path)
    toolbox(FakeGitHub({PUBSPEC: file_entry(text)}), cache).call(
        "read_file", {"path": "pubspec.yaml"}
    )

    small = ToolBox(ToolEnv(context(), FakeGitHub()), cache=cache, cap_tokens=200)
    hit = small.call("read_file", {"path": "pubspec.yaml"})

    assert hit.cached
    assert hit.text.endswith(TRUNCATED)
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["tool"] == "read_file"
    assert len(stored["output"]) > len(hit.text)


def test_unreadable_entries_are_misses(tmp_path: Path) -> None:
    cache = ToolCache(tmp_path)
    key = tool_cache_key("o/a", "commit:x", "read_file", {})
    (tmp_path / f"{key}.json").write_text("{not json")

    assert cache.get(key) is None


def test_an_unwritable_cache_never_breaks_a_tool_call(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    cache = ToolCache(blocker / "cache")  # its parent is a file: mkdir fails
    github = FakeGitHub({PUBSPEC: file_entry("name: app\n")})

    out = toolbox(github, cache).call("read_file", {"path": "pubspec.yaml"})

    assert out.ok
