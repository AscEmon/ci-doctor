import pytest

from cidoc.miner.rules import is_fork_run, workflow_skip_reason


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
