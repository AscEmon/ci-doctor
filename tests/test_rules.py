from pathlib import Path

import pytest

from cidoc.miner.rules import (
    EXCLUDE_FILENAME,
    RepoExcludes,
    is_fork_run,
    load_repo_excludes,
    parse_exclude_file,
    workflow_skip_reason,
)


@pytest.mark.parametrize(
    ("path", "name", "expected"),
    [
        (".github/workflows/ci.yml", "CI", None),
        (".github/workflows/build.yml", "Build & Test", None),
        ("dynamic/pages/pages-build-deployment", "pages build and deployment", "dynamic workflow"),
        (".github/workflows/deps.yml", "Dependabot auto-merge", "dependabot workflow"),
        (".github/workflows/codeql-analysis.yml", "CodeQL", "codeql workflow"),
        (".github/workflows/x.yml", "copilot-setup-steps", "copilot workflow"),
    ],
)
def test_workflow_skip_reason(path: str, name: str, expected: str | None) -> None:
    assert workflow_skip_reason(path, name) == expected


@pytest.mark.parametrize(
    ("head_repository", "expected"),
    [
        ({"full_name": "acme/app"}, False),
        ({"full_name": "Acme/App"}, False),
        ({"full_name": "someone/app"}, True),
        (None, True),
    ],
)
def test_is_fork_run(head_repository: dict[str, str] | None, expected: bool) -> None:
    assert is_fork_run({"head_repository": head_repository}, "acme/app") is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("flutter/flutter", "excluded org flutter/*"),
        ("Flutter/Engine", "excluded org flutter/*"),
        ("dart-lang/sdk", "excluded org dart-lang/*"),
        ("acme/flutter", None),
        ("flutter-community/plus_plugins", None),
    ],
)
def test_default_excludes(name: str, expected: str | None) -> None:
    assert RepoExcludes().reason(name) == expected


def test_parse_exclude_file() -> None:
    text = "# extra\nAcme/Big-Monorepo\n\nsome-org/*  # whole org\nacme/big-monorepo\n"

    owners, repos = parse_exclude_file(text)

    assert owners == {"some-org"}
    assert repos == {"acme/big-monorepo"}


@pytest.mark.parametrize("line", ["just-a-name", "*/repo", "o/r/extra", "https://github.com/o/r"])
def test_parse_exclude_file_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match=r"exclude_repos\.txt line 2"):
        parse_exclude_file(f"ok/repo\n{line}\n")


def test_load_repo_excludes_without_file_uses_defaults(tmp_path: Path) -> None:
    assert load_repo_excludes(tmp_path) == RepoExcludes()


def test_load_repo_excludes_adds_file_entries_to_defaults(tmp_path: Path) -> None:
    (tmp_path / EXCLUDE_FILENAME).write_text("acme/huge\nother-org/*\n", encoding="utf-8")

    excludes = load_repo_excludes(tmp_path)

    assert excludes.reason("flutter/packages") == "excluded org flutter/*"
    assert excludes.reason("ACME/huge") == "listed in exclude_repos.txt"
    assert excludes.reason("other-org/x") == "excluded org other-org/*"
    assert excludes.reason("acme/small") is None
