"""Typer entrypoint for the `cidoc` command."""

from importlib.metadata import version

import typer
from rich.console import Console

app = typer.Typer(
    help="ci-doctor: diagnose CI failures from GitHub Actions logs.", no_args_is_help=True
)
console = Console()


@app.callback()
def main() -> None:
    """ci-doctor command line interface."""


@app.command("version")
def show_version() -> None:
    """Print the installed ci-doctor version."""
    console.print(version("ci-doctor"))
