"""Typer entrypoint for the `cidoc` command."""

import random
import sqlite3
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.github_client import GitHubClient
from cidoc.labeling.labels import LABELS_FILENAME, load_labels
from cidoc.labeling.session import load_candidates, pick_stratified, run_session
from cidoc.miner.discover import RepoInfo, discover
from cidoc.miner.runs import FailedJob, mine_repo, repos_to_mine
from cidoc.miner.seeds import SEED_FILENAME, add_seed_repos
from cidoc.miner.stats import collect_stats, pct

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
    seed_file: Annotated[
        Path | None,
        typer.Option(help=f"owner/repo list to add as-is (default: <data_dir>/{SEED_FILENAME})."),
    ] = None,
    search: Annotated[bool, typer.Option(help="Run the GitHub search step.")] = True,
) -> None:
    """Find Dart repos with Flutter workflows and store them in the repos table."""

    def report(repo: RepoInfo, kept: bool) -> None:
        mark = "[green]keep[/green]" if kept else "[dim]skip[/dim]"
        console.print(f"{mark} {repo.full_name} ({repo.stars}★)")

    def report_missing(name: str) -> None:
        console.print(f"[yellow]seed not found[/yellow] {name}")

    seed_path = seed_file or get_settings().data_dir / SEED_FILENAME
    if seed_file and not seed_file.exists():
        raise typer.BadParameter(f"{seed_file} does not exist", param_hint="--seed-file")

    now = datetime.now(UTC)
    conn = open_db()
    try:
        with open_github() as client:
            if seed_path.exists():
                seeded = add_seed_repos(client, conn, seed_path, now, on_missing=report_missing)
                console.print(f"Added {seeded} seed repos from {seed_path}.")
            if search:
                added = discover(client, conn, now, limit, on_checked=report)
                console.print(f"Added {added} repos from search.")
    finally:
        conn.close()


@mine_app.command("runs")
def mine_runs(
    max_per_repo: Annotated[
        int, typer.Option(help="Max failed runs to mine per repo (last 90 days).")
    ] = 30,
    repo: Annotated[
        list[str] | None, typer.Option(help="Only mine these owner/repo names (repeatable).")
    ] = None,
) -> None:
    """Download logs for failed jobs of every stored repo into the failures table."""
    data_dir = get_settings().data_dir
    now = datetime.now(UTC)

    def report(job: FailedJob | None, job_id: int) -> None:
        if job is None:
            console.print(f"  [yellow]log expired[/yellow] job {job_id}")
            return
        steps = ", ".join(job.failed_steps) or "?"
        console.print(f"  job {job_id} [{job.runner_os or '?'}] {job.workflow_name}: {steps}")

    conn = open_db()
    try:
        targets = repo or repos_to_mine(conn)
        with open_github() as client:
            for name in targets:
                console.print(f"[bold]{name}[/bold]")
                stored = mine_repo(client, conn, data_dir, name, now, max_per_repo, report)
                console.print(f"  stored {stored} new failed jobs")
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow] Progress is saved; rerun to resume.")
        raise typer.Exit(130) from None
    finally:
        conn.close()


def count_table(title: str, label: str, rows: list[tuple[str, int]], total: int) -> Table:
    table = Table(title=title)
    table.add_column(label)
    table.add_column("jobs", justify="right")
    table.add_column("%", justify="right")
    for name, count in rows:
        table.add_row(name, str(count), f"{pct(count, total):.1f}")
    return table


@mine_app.command("stats")
def mine_stats(
    top_steps: Annotated[int, typer.Option(help="How many failed step names to list.")] = 20,
) -> None:
    """Summarize mined failures per repo, runner OS, and failed step."""
    conn = open_db()
    try:
        stats = collect_stats(conn, top_steps=top_steps)
    finally:
        conn.close()

    console.print(
        f"{stats.jobs} failed jobs across {stats.repos} repos; "
        f"{stats.with_fix} ({pct(stats.with_fix, stats.jobs):.1f}%) have a fix_sha; "
        f"{stats.redactions} secrets redacted."
    )

    repos = Table(title="Per repo")
    repos.add_column("repo")
    repos.add_column("jobs", justify="right")
    repos.add_column("with fix_sha", justify="right")
    for r in stats.per_repo:
        repos.add_row(r.repo, str(r.jobs), f"{r.with_fix} ({pct(r.with_fix, r.jobs):.0f}%)")
    console.print(repos)
    console.print(count_table("Per runner OS", "runner_os", stats.per_runner_os, stats.jobs))
    console.print(
        count_table(f"Top {top_steps} failed steps", "step", stats.per_failed_step, stats.jobs)
    )


@app.command("label")
def label(
    n: Annotated[int, typer.Option("--n", help="How many failures to label this session.")] = 50,
    seed: Annotated[int | None, typer.Option(help="Random seed for a reproducible pick.")] = None,
) -> None:
    """Label unlabeled failures by hand; answers are appended to labels.jsonl."""
    data_dir = get_settings().data_dir
    labels_path = data_dir / LABELS_FILENAME

    def warn_bad_line(lineno: int) -> None:
        console.print(f"[yellow]ignoring unreadable line {lineno} in {labels_path}[/yellow]")

    existing = load_labels(labels_path, on_bad_line=warn_bad_line)
    conn = open_db()
    try:
        candidates = load_candidates(conn, exclude={lbl.job_id for lbl in existing})
    finally:
        conn.close()
    rng = random.Random(seed)  # noqa: S311 - sampling order only, not security
    picks = pick_stratified(candidates, [lbl.repo for lbl in existing], n, rng)
    if not picks:
        console.print("Nothing left to label.")
        return

    console.print(
        f"{len(existing)} labeled so far; {len(candidates)} unlabeled; picked {len(picks)}."
    )
    result = run_session(
        console,
        picks,
        data_dir,
        labels_path,
        getch=typer.getchar,
        ask=console.input,
        clock=lambda: datetime.now(UTC),
    )
    status = "Stopped" if result.quit else "Done"
    console.print(
        f"{status}: {result.labeled} labeled, {result.skipped} skipped "
        f"({len(existing) + result.labeled} total in {labels_path}). Run again to resume."
    )
