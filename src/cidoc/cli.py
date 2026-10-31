"""Typer entrypoint for the `cidoc` command."""

import sqlite3
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Annotated

import typer
from rich.console import Console

from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.github_client import GitHubClient
from cidoc.miner.discover import RepoInfo, discover

app = typer.Typer(
    help="ci-doctor: diagnose CI failures from GitHub Actions logs.", no_args_is_help=True
)
mine_app = typer.Typer(help="Collect failed GitHub Actions runs.", no_args_is_help=True)
app.add_typer(mine_app, name="mine")
console = Console()


def open_db() -> sqlite3.Connection:
    return connect(get_settings().data_dir / DB_FILENAME)


def open_github() -> GitHubClient:
    settings = get_settings()
    return GitHubClient(settings.github_token, base_url=settings.github_api_url)


@app.callback()
def main() -> None:
    """ci-doctor command line interface."""


@app.command("version")
def show_version() -> None:
    """Print the installed ci-doctor version."""
    console.print(version("ci-doctor"))


@mine_app.command("discover")
def mine_discover(
    limit: Annotated[int, typer.Option(help="Max search results to check (API cap: 1000).")] = 300,
) -> None:
    """Find Dart repos with Flutter workflows and store them in the repos table."""

    def report(repo: RepoInfo, kept: bool) -> None:
        mark = "[green]keep[/green]" if kept else "[dim]skip[/dim]"
        console.print(f"{mark} {repo.full_name} ({repo.stars}★)")

    conn = open_db()
    try:
        with open_github() as client:
            added = discover(client, conn, datetime.now(UTC), limit, on_checked=report)
    finally:
        conn.close()
    console.print(f"Added {added} repos.")
