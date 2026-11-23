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
from rich.text import Text

from cidoc.config import get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.github_client import GitHubClient
from cidoc.labeling.labels import LABELS_FILENAME, load_labels
from cidoc.labeling.session import cap_per_run, load_candidates, pick_stratified, run_session
from cidoc.miner.discover import RepoInfo, discover
from cidoc.miner.prune import apply_prune, plan_prune
from cidoc.miner.redaction_report import (
    CONTEXT_CHARS,
    EXAMPLES_PER_RULE,
    collect_redaction_report,
)
from cidoc.miner.rules import RepoExcludes, load_repo_excludes
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


def load_excludes() -> RepoExcludes:
    try:
        return load_repo_excludes(get_settings().data_dir)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


def report_excluded(name: str, reason: str) -> None:
    console.print(f"[dim]excluded[/dim] {name} ({reason})")


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

    excludes = load_excludes()
    now = datetime.now(UTC)
    conn = open_db()
    try:
        with open_github() as client:
            if seed_path.exists():
                seeded = add_seed_repos(
                    client,
                    conn,
                    seed_path,
                    now,
                    on_missing=report_missing,
                    excludes=excludes,
                    on_excluded=report_excluded,
                )
                console.print(f"Added {seeded} seed repos from {seed_path}.")
            if search:
                added = discover(
                    client,
                    conn,
                    now,
                    limit,
                    on_checked=report,
                    excludes=excludes,
                    on_excluded=report_excluded,
                )
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
    excludes = load_excludes()
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
                reason = excludes.reason(name)
                if reason:
                    report_excluded(name, reason)
                    continue
                console.print(f"[bold]{name}[/bold]")
                result = mine_repo(client, conn, data_dir, name, now, max_per_repo, report)
                console.print(f"  stored {result.stored} new failed jobs")
                if result.skipped_runs:
                    console.print(
                        f"  [yellow]skipped {result.skipped_runs} runs after errors[/yellow]"
                        " (rerun to retry them)"
                    )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow] Progress is saved; rerun to resume.")
        raise typer.Exit(130) from None
    finally:
        conn.close()


@mine_app.command("prune")
def mine_prune(
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually delete. Without it, only report.")
    ] = False,
) -> None:
    """Remove stored failures and logs that the mining rules now exclude (dry-run by default)."""
    data_dir = get_settings().data_dir
    excludes = load_excludes()
    conn = open_db()
    try:
        plan = plan_prune(conn, excludes)
        if plan.empty:
            console.print("Nothing to prune.")
            return

        verb = "Removing" if apply else "Would remove"
        table = Table(title=f"{verb} {len(plan.failures)} failed jobs")
        table.add_column("reason")
        table.add_column("jobs", justify="right")
        for reason, count in plan.counts_by_reason():
            table.add_row(reason, str(count))
        console.print(table)
        if plan.repos:
            console.print(f"{verb} {len(plan.repos)} excluded repos from the repos table:")
            for name, reason in plan.repos:
                console.print(f"  {name} ({reason})")

        labeled = {lbl.job_id for lbl in load_labels(data_dir / LABELS_FILENAME)}
        pruned_labeled = sum(f.job_id in labeled for f in plan.failures)
        if pruned_labeled:
            console.print(
                f"[yellow]{pruned_labeled} of these jobs are in {LABELS_FILENAME}.[/yellow]"
                " Their labels are kept; they will no longer match a stored failure."
            )

        if not apply:
            console.print("Dry run: nothing deleted. Rerun with --apply to delete.")
            return
        result = apply_prune(conn, data_dir, plan)
    finally:
        conn.close()
    console.print(
        f"Deleted {result.failures} failures, {result.log_files} log files, "
        f"{result.mined_runs} mined-run markers, {result.repos} repos."
    )


@mine_app.command("redaction-report")
def mine_redaction_report(
    examples: Annotated[
        int, typer.Option(help="Random masked examples to show per rule.")
    ] = EXAMPLES_PER_RULE,
    context: Annotated[
        int, typer.Option(help="Characters of log context either side of each mask.")
    ] = CONTEXT_CHARS,
    seed: Annotated[int | None, typer.Option(help="Random seed for reproducible examples.")] = None,
) -> None:
    """Audit redactions: counts per rule and masked examples in context (never unmasked)."""
    data_dir = get_settings().data_dir
    conn = open_db()
    try:
        rng = random.Random(seed)  # noqa: S311 - sampling order only, not security
        report = collect_redaction_report(conn, data_dir, rng, examples, context)
    finally:
        conn.close()

    if not report.rules:
        console.print("No redactions in stored logs.")
    else:
        table = Table(title=f"Redactions in {report.logs_with_redactions} logs")
        table.add_column("rule")
        table.add_column("redactions", justify="right")
        table.add_column("logs", justify="right")
        for rule in report.rules:
            table.add_row(rule.rule, str(rule.redactions), str(rule.logs))
        console.print(table)
    if report.logs_counted_from_markers:
        console.print(
            f"[dim]{report.logs_counted_from_markers} logs predate per-rule tracking;"
            " their counts come from the masks in the stored log.[/dim]"
        )
    if report.logs_missing:
        console.print(
            f"[yellow]{len(report.logs_missing)} log files are missing[/yellow] and were skipped."
        )

    for rule in report.rules:
        console.print(f"\n[bold]{rule.rule}[/bold] ({len(rule.examples)} examples)")
        for ex in rule.examples:
            # Logs are untrusted text: print without markup so [REDACTED:...] and any
            # bracketed log content render literally.
            console.print(
                Text.assemble((f"  {ex.repo} job {ex.job_id} L{ex.line_no}: ", "dim"), ex.snippet)
            )


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
    top_dups: Annotated[
        int, typer.Option(help="How many likely matrix-duplicate runs to list.")
    ] = 10,
) -> None:
    """Summarize mined failures per repo, runner OS, failed step, and run."""
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
    console.print(
        f"{stats.runs} unique runs; {stats.jobs_per_run:.2f} failed jobs per run on average."
    )
    if stats.matrix_duplicates:
        console.print(
            f"[yellow]{len(stats.matrix_duplicates)} runs have likely matrix duplicates"
            f"[/yellow] (several jobs failing at the same first step): "
            f"{stats.duplicate_jobs} jobs ({pct(stats.duplicate_jobs, stats.jobs):.1f}%) "
            "are repeats. `cidoc label` samples one job per run by default."
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
    if stats.matrix_duplicates:
        dups = Table(title=f"Likely matrix duplicates (top {top_dups})")
        dups.add_column("repo")
        dups.add_column("run_id", justify="right")
        dups.add_column("first failed step")
        dups.add_column("jobs", justify="right")
        for d in stats.matrix_duplicates[:top_dups]:
            dups.add_row(d.repo, str(d.run_id), d.step, str(d.jobs))
        console.print(dups)


@app.command("label")
def label(
    n: Annotated[int, typer.Option("--n", help="How many failures to label this session.")] = 50,
    seed: Annotated[int | None, typer.Option(help="Random seed for a reproducible pick.")] = None,
    max_per_run: Annotated[
        int,
        typer.Option(
            min=0,
            help="Max jobs to label per workflow run, counting earlier labels (0 = no limit).",
        ),
    ] = 1,
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
    unlabeled = len(candidates)
    if max_per_run:
        labeled_runs = [(lbl.repo, lbl.run_id) for lbl in existing]
        candidates = cap_per_run(candidates, labeled_runs, max_per_run, rng)
    picks = pick_stratified(candidates, [lbl.repo for lbl in existing], n, rng)
    if not picks:
        console.print("Nothing left to label.")
        return

    capped = ""
    if len(candidates) < unlabeled:
        capped = f" ({len(candidates)} eligible at {max_per_run} per run)"
    console.print(
        f"{len(existing)} labeled so far; {unlabeled} unlabeled{capped}; picked {len(picks)}."
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
