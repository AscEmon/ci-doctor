"""Typer entrypoint for the `cidoc` command."""

import random
import sqlite3
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from cidoc.agent.budget import BudgetedToolProvider
from cidoc.agent.display import render_agent_input, render_evidence_files, render_trajectory
from cidoc.agent.loop import DEFAULT_MAX_STEPS, run_agent_cached
from cidoc.agent.runtime import (
    agent_setup,
    job_context,
    make_toolbox,
    open_tools_github,
    parse_tool_names,
)
from cidoc.agent.tools import TOOLS, UnknownToolError
from cidoc.analyze.cache import ResponseCache, analyze_cached
from cidoc.analyze.costguard import (
    BudgetedProvider,
    BudgetExceededError,
    CostGuard,
    estimate_run,
)
from cidoc.analyze.display import render_input, render_outputs, render_result, render_usage
from cidoc.analyze.inputs import AnalysisInput, InputError, resolve_input
from cidoc.analyze.preprocess import preprocess
from cidoc.analyze.prompt import GuidelineError, Prompt, load_prompt
from cidoc.config import MissingGeminiKeyError, get_settings
from cidoc.db import DB_FILENAME, connect
from cidoc.eval.baseline import REGEX_BASELINE, Predictor, RegexBaseline
from cidoc.eval.compare import compare_runs
from cidoc.eval.display import (
    render_comparison,
    render_estimate,
    render_item,
    render_items,
    render_summary,
    select_items,
)
from cidoc.eval.metrics import compute_metrics
from cidoc.eval.results import RunInfo, load_run, run_path, write_run
from cidoc.eval.runner import (
    ALL_SPLITS,
    AgentPredictor,
    LLMPredictor,
    load_cases,
    plan_agent_run,
    plan_run,
    run_eval,
)
from cidoc.github_client import GitHubClient
from cidoc.labeling.labels import (
    CATEGORIES,
    DEV_SPLIT,
    LABELS_FILENAME,
    SPLITS,
    TEST2_SPLIT,
    LabelNotFoundError,
    StoredLabel,
    find_label,
    load_labels,
    newest_labels,
    replace_label_line,
)
from cidoc.labeling.session import (
    Candidate,
    QuitRequestedError,
    ViewToggle,
    cap_per_run,
    count_unlabeled_over_redacted,
    edit_one,
    load_candidate,
    load_candidates,
    pick_stratified,
    render_header,
    run_session,
)
from cidoc.llm.base import LLMError
from cidoc.llm.factory import create_provider, create_tool_provider
from cidoc.logview import build_focus_view, build_view, clean_log, read_log
from cidoc.miner.discover import RepoInfo, discover
from cidoc.miner.prune import apply_prune, plan_prune
from cidoc.miner.redaction_report import (
    CONTEXT_CHARS,
    EXAMPLES_PER_RULE,
    RedactionReport,
    build_redaction_report,
    collect_redaction_report,
)
from cidoc.miner.refetch import (
    RefetchTarget,
    count_over_redacted,
    find_refetch_targets,
    preview_refetch,
    refetch_logs,
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


@app.command("eval")
def eval_command(
    model: Annotated[
        str | None,
        typer.Option(help=f"LLM model, or {REGEX_BASELINE} (default: CIDOC_MODEL)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Only the first N labels (in labels.jsonl order).")
    ] = None,
    split: Annotated[
        str,
        typer.Option(help="Which labels to score: dev, test1, test2 or all (missing = dev)."),
    ] = DEV_SPLIT,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore cached results and call the API again.")
    ] = False,
    max_cost: Annotated[
        float | None,
        typer.Option(
            "--max-cost",
            min=0,
            help="Stop before a call whose estimated cost would push the run past this USD.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Estimate tokens, cost and duration; make no API calls."),
    ] = False,
    compare: Annotated[
        tuple[Path, Path] | None,
        typer.Option(
            "--compare",
            metavar="A.json B.json",
            help="Compare two saved runs (metric deltas, disagreements) instead of running.",
        ),
    ] = None,
    show: Annotated[
        Path | None,
        typer.Option(
            "--show",
            metavar="RUN.json",
            help="Print a saved run item by item (label next to prediction) instead of running.",
        ),
    ] = None,
    wrong: Annotated[
        bool, typer.Option("--wrong", help="With --show: only items predicted wrong.")
    ] = False,
    job: Annotated[int | None, typer.Option("--job", help="With --show: only this job id.")] = None,
    agent: Annotated[
        bool,
        typer.Option("--agent", help="Score the Phase 2 agent (tools before answering)."),
    ] = False,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            min=1,
            help=f"With --agent: tool calls allowed per item (default {DEFAULT_MAX_STEPS}).",
        ),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools", metavar="a,b,...", help="With --agent: only these tools (default all)."
        ),
    ] = None,
) -> None:
    """Score a model against the human labels and save the run to data/eval_runs/.

    Resumable: items already in the response cache cost nothing. An item whose API call
    still fails after retries is recorded as not scored and the run goes on; a re-run
    retries it. Only --max-cost and Ctrl-C stop the run early, after saving what it
    scored. Exit code 1 if anything was not scored, 130 after Ctrl-C.
    """
    settings = get_settings()
    if split not in (*SPLITS, ALL_SPLITS):
        console.print(f"[red]--split must be one of {', '.join((*SPLITS, ALL_SPLITS))}.[/red]")
        raise typer.Exit(2)
    if show is not None and compare is not None:
        console.print("[red]Use --show or --compare, not both.[/red]")
        raise typer.Exit(2)
    if (wrong or job is not None) and show is None:
        console.print("[red]--wrong and --job only apply to --show.[/red]")
        raise typer.Exit(2)
    if not agent and (max_steps is not None or tools is not None):
        console.print("[red]--max-steps and --tools only apply to --agent.[/red]")
        raise typer.Exit(2)
    try:
        tool_names = parse_tool_names(tools)
    except UnknownToolError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(2) from None
    if show is not None:
        try:
            saved = load_run(show)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            console.print(Text(f"cannot read run file: {exc}", style="red"))
            raise typer.Exit(2) from None
        if job is not None and all(item.job_id != job for item in saved.items):
            console.print(Text(f"job {job} is not in {show}", style="red"))
            raise typer.Exit(2)
        items = select_items(saved.items, wrong, job)
        root_causes = {
            label.job_id: label.root_cause
            for label in load_labels(settings.data_dir / LABELS_FILENAME)
        }
        render_items(console, saved, items, root_causes)
        unscored = sum(item.error is not None for item in saved.items)
        wrong_total = sum(item.error is None and not item.correct for item in saved.items)
        console.print(
            Text(
                f"Showing {len(items)} of {len(saved.items)} items"
                f" ({wrong_total} wrong, {unscored} not scored).",
                style="dim",
            )
        )
        return
    if compare is not None:
        try:
            run_a, run_b = load_run(compare[0]), load_run(compare[1])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            console.print(Text(f"cannot read run file: {exc}", style="red"))
            raise typer.Exit(2) from None
        labels = newest_labels(load_labels(settings.data_dir / LABELS_FILENAME))
        comparison = compare_runs(run_a, run_b, settings.price_for, labels)
        render_comparison(console, run_a, run_b, comparison)
        return
    try:
        prompt = load_prompt()
    except GuidelineError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(2) from None
    name = model or settings.cidoc_model
    baseline = name == REGEX_BASELINE
    if baseline and agent:
        console.print(f"[red]--agent needs an LLM model, not {REGEX_BASELINE}.[/red]")
        raise typer.Exit(2)
    setup = agent_setup(prompt, tool_names, max_steps) if agent else None
    price = settings.price_for(name)
    if max_cost is not None and not baseline and price is None:
        console.print(Text(f"--max-cost needs a price for {name}: set CIDOC_PRICES.", "red"))
        raise typer.Exit(2)

    loaded = load_cases(settings.data_dir, open_db, limit, split)
    if loaded.missing:
        console.print(
            f"[yellow]{len(loaded.missing)} labeled jobs have no log and are not scored:[/yellow]"
            f" {', '.join(map(str, loaded.missing))}"
        )
    if not loaded.cases:
        console.print(f"[red]No labeled jobs with logs to evaluate in the {split} split.[/red]")
        raise typer.Exit(1)

    budget = settings.cidoc_token_budget
    cache = ResponseCache.in_data_dir(settings.data_dir)
    if dry_run:
        if baseline:
            console.print(
                f"dry run: {REGEX_BASELINE}, {split} split, {len(loaded.cases)} items,"
                " no API calls, $0"
            )
            return
        if setup is not None:
            plan = plan_agent_run(
                loaded.cases, setup, cache, name, budget, open_db, refresh=refresh
            )
        else:
            plan = plan_run(loaded.cases, prompt, cache, name, budget, refresh=refresh)
        estimate = estimate_run(plan.calls, plan.cached, price, settings.cidoc_rpm)
        render_estimate(
            console,
            name,
            estimate,
            settings.cidoc_rpm,
            plan.calls,
            price,
            max_cost,
            max_steps=setup.limits.max_steps if setup else None,
        )
        return

    predictor: Predictor
    github: GitHubClient | None = None
    if baseline:
        predictor, prompt_version, run_budget = RegexBaseline(), None, None
    elif setup is not None:
        try:
            tool_provider = create_tool_provider(settings, name)
        except MissingGeminiKeyError as exc:
            console.print(Text(str(exc), style="red"))
            raise typer.Exit(2) from None
        if max_cost is not None and price is not None:
            tool_provider = BudgetedToolProvider(tool_provider, CostGuard(max_cost, price))
        github = open_tools_github(settings.github_token, settings.github_api_url)
        predictor = AgentPredictor(
            tool_provider,
            setup,
            cache,
            budget,
            settings.data_dir,
            open_db,
            github,
            price=price,
            refresh=refresh,
        )
        prompt_version, run_budget = setup.prompt.version, budget
    else:
        try:
            provider = create_provider(settings, name)
        except MissingGeminiKeyError as exc:
            console.print(Text(str(exc), style="red"))
            raise typer.Exit(2) from None
        if max_cost is not None and price is not None:
            provider = BudgetedProvider(provider, CostGuard(max_cost, price))
        predictor = LLMPredictor(provider, prompt, cache, budget, refresh=refresh)
        prompt_version, run_budget = prompt.version, budget

    total = len(loaded.cases)
    try:
        outcome = run_eval(
            loaded.cases,
            predictor,
            on_item=lambda n, item: render_item(console, n, total, item),
            stop_on=(BudgetExceededError,),
        )
    finally:
        if github is not None:
            github.close()
    metrics = compute_metrics(outcome.items, price)
    info = RunInfo(
        model=predictor.name,
        created_at=datetime.now(UTC).isoformat(),
        prompt_version=prompt_version,
        guideline_hash=prompt.guideline_hash,
        token_budget=run_budget,
        labels_total=loaded.labels_total,
        complete=outcome.stopped is None,
        stopped=outcome.stopped,
        missing=loaded.missing,
        split=split,
        agent=setup is not None,
        tools=setup.tools if setup else None,
        max_steps=setup.limits.max_steps if setup else None,
    )
    path = run_path(settings.data_dir, predictor.name, datetime.now(UTC), agent=info.agent)
    write_run(path, info, outcome.items, metrics)
    render_summary(console, info, metrics)
    if outcome.stopped:
        console.print(
            Text(
                f"Stopped after {len(outcome.items)} of {total}: {outcome.stopped}\n"
                "Saved the partial run; run the same command again to resume from the cache.",
                style="yellow",
            )
        )
    if outcome.errored:
        console.print(
            Text(
                f"{outcome.errored} items failed after retries and were not scored"
                " (left out of the metrics, not cached); run again to retry them.",
                style="yellow",
            )
        )
    console.print(f"Saved {path}")
    if outcome.interrupted:
        raise typer.Exit(130)
    if outcome.stopped or outcome.errored:
        raise typer.Exit(1)


@app.command("analyze")
def analyze(
    target: Annotated[
        str, typer.Argument(help="A job log file (.log.gz or text), or a stored job id.")
    ],
    model: Annotated[
        str | None, typer.Option(help="LLM model to use (default: CIDOC_MODEL).")
    ] = None,
    budget: Annotated[
        int | None,
        typer.Option(min=1, help="Token budget for the log excerpt (default: CIDOC_TOKEN_BUDGET)."),
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore a cached result and call the API again.")
    ] = False,
    max_cost: Annotated[
        float | None,
        typer.Option(
            "--max-cost",
            min=0,
            help="Stop before a call whose estimated cost would exceed this many USD.",
        ),
    ] = None,
    show_input: Annotated[
        bool,
        typer.Option(
            "--show-input",
            help="Print the exact system instruction and log sent, then the raw model output.",
        ),
    ] = False,
    no_call: Annotated[
        bool,
        typer.Option(
            "--no-call", help="Print what would be sent and stop: no API call, no key needed."
        ),
    ] = False,
    agent: Annotated[
        bool,
        typer.Option(
            "--agent",
            help="Phase 2: let the model call read-only tools (repo files, run context,"
            " full log) before it answers. Needs a stored job id.",
        ),
    ] = False,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            min=1,
            help=f"With --agent: tool calls allowed (default {DEFAULT_MAX_STEPS}).",
        ),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            metavar="a,b,...",
            help=f"With --agent: only these tools (default all: {','.join(TOOLS)}).",
        ),
    ] = None,
) -> None:
    """Diagnose one failed job with the LLM."""
    settings = get_settings()
    if not agent and (max_steps is not None or tools is not None):
        console.print("[red]--max-steps and --tools only apply to --agent.[/red]")
        raise typer.Exit(2)
    try:
        source = resolve_input(target, settings.data_dir, open_db)
        prompt = load_prompt()
        tool_names = parse_tool_names(tools)
    except (InputError, GuidelineError, UnknownToolError) as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(2) from None
    name = model or settings.cidoc_model
    if agent:
        analyze_agent(
            source, prompt, name, budget, refresh, max_cost, show_input, no_call, tool_names,
            max_steps,
        )  # fmt: skip
        return
    if source.redactions:
        console.print(f"[yellow]Masked {source.redactions} secret-looking values before sending.")
    if source.renumbered:
        console.print("[yellow]Redaction joined lines: line numbers refer to the redacted log.")

    log = preprocess(source.text, source.failed_steps, budget or settings.cidoc_token_budget)
    console.print(
        Text(
            f"{source.source}: {log.total_lines} lines, ~{log.estimated_tokens} tokens sent"
            f" (prompt {prompt.version}, model {name})",
            style="dim",
        )
    )
    if show_input or no_call:
        render_input(console, prompt, log, name)
    if no_call:
        return

    try:
        provider = create_provider(settings, model)
    except MissingGeminiKeyError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(2) from None
    price = settings.price_for(provider.model)
    if max_cost is not None:
        if price is None:
            console.print(
                Text(f"--max-cost needs a price for {provider.model}: set CIDOC_PRICES.", "red")
            )
            raise typer.Exit(2)
        provider = BudgetedProvider(provider, CostGuard(max_cost, price))
    try:
        cache = ResponseCache.in_data_dir(settings.data_dir)
        result = analyze_cached(provider, log, prompt, cache, refresh=refresh)
    except BudgetExceededError as exc:
        console.print(Text(f"Stopped: {exc}", style="yellow"))
        raise typer.Exit(1) from None
    except LLMError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from None
    if show_input:
        render_outputs(console, result)
    render_result(console, result, clean_log(source.text))
    render_usage(console, result, price)
    if not result.ok:
        raise typer.Exit(1)


def analyze_agent(
    source: AnalysisInput,
    base: Prompt,
    name: str,
    budget: int | None,
    refresh: bool,
    max_cost: float | None,
    show_input: bool,
    no_call: bool,
    tool_names: list[str] | None,
    max_steps: int | None,
) -> None:
    """`cidoc analyze --agent`: run the agent and print its trajectory and diagnosis."""
    settings = get_settings()
    if source.job_id is None:
        console.print(
            Text("--agent needs a stored job id: its tools read the job's run and repo.", "red")
        )
        raise typer.Exit(2)
    setup = agent_setup(base, tool_names, max_steps)
    log = preprocess(source.text, source.failed_steps, budget or settings.cidoc_token_budget)
    console.print(
        Text(
            f"{source.source}: {log.total_lines} lines, ~{log.estimated_tokens} tokens of"
            f" excerpt (agent prompt {setup.prompt.version}, model {name},"
            f" tools {','.join(setup.tools)}, max {setup.limits.max_steps} steps)",
            style="dim",
        )
    )
    if show_input or no_call:
        render_agent_input(console, setup.prompt, log, name)
    if no_call:
        return
    try:
        provider = create_tool_provider(settings, name)
    except MissingGeminiKeyError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(2) from None
    price = settings.price_for(provider.model)
    if max_cost is not None:
        if price is None:
            console.print(
                Text(f"--max-cost needs a price for {provider.model}: set CIDOC_PRICES.", "red")
            )
            raise typer.Exit(2)
        provider = BudgetedToolProvider(provider, CostGuard(max_cost, price))
    ctx = job_context(source.job_id, source.text, open_db)
    github = open_tools_github(settings.github_token, settings.github_api_url)
    try:
        toolbox = make_toolbox(ctx, github, setup, settings.data_dir)
        result = run_agent_cached(
            provider,
            log,
            setup.prompt,
            toolbox,
            ResponseCache.in_data_dir(settings.data_dir),
            setup.limits,
            price=price,
            refresh=refresh,
        )
    except BudgetExceededError as exc:
        console.print(Text(f"Stopped: {exc}", style="yellow"))
        raise typer.Exit(1) from None
    except LLMError as exc:
        console.print(Text(str(exc), style="red"))
        raise typer.Exit(1) from None
    finally:
        github.close()
    if show_input:
        render_outputs(console, result)
    if result.trajectory is not None:
        render_trajectory(console, result.trajectory, cached=result.cached)
    console.rule("diagnosis", style="cyan")
    render_result(console, result, ctx.log_lines)
    render_evidence_files(console, result)
    if not result.ok:
        raise typer.Exit(1)


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

    def report_excluded_job(job_id: int, reason: str) -> None:
        console.print(f"  [dim]out of scope[/dim] job {job_id}: {reason}")

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
                result = mine_repo(
                    client,
                    conn,
                    data_dir,
                    name,
                    now,
                    max_per_repo,
                    report,
                    on_excluded=report_excluded_job,
                )
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
        plan = plan_prune(conn, excludes, data_dir)
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

    print_redaction_report(report, empty="No redactions in stored logs.")


def print_redaction_report(report: RedactionReport, empty: str) -> None:
    """Counts per rule, then masked examples. Shared by redaction-report and refetch preview."""
    if not report.rules:
        console.print(empty)
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


@mine_app.command("refetch")
def mine_refetch(
    only_redacted: Annotated[
        bool,
        typer.Option(
            "--only-redacted",
            help="Refetch logs that older rules masked with a generic (heuristic) rule.",
        ),
    ] = False,
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually download. Without it, only report.")
    ] = False,
    preview: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Download N random targets, redact in memory and report. Writes nothing.",
        ),
    ] = None,
    examples: Annotated[
        int, typer.Option(help="Preview: masked examples to show per rule.")
    ] = EXAMPLES_PER_RULE,
    seed: Annotated[int | None, typer.Option(help="Preview: random seed for the sample.")] = None,
) -> None:
    """Re-download over-redacted logs and redact them with the current rules (dry-run default)."""
    if not only_redacted:
        console.print("[red]Only --only-redacted is supported for now.[/red]")
        raise typer.Exit(2)
    if preview is not None and apply:
        console.print("[red]--preview writes nothing; drop --apply to preview.[/red]")
        raise typer.Exit(2)
    data_dir = get_settings().data_dir
    conn = open_db()
    try:
        targets = find_refetch_targets(conn, data_dir)
        already = count_over_redacted(conn)
        if already:
            console.print(
                f"[dim]{already} logs are already marked over_redacted (expired on GitHub);"
                " skipped.[/dim]"
            )
        if not targets:
            console.print("Nothing to refetch.")
            return
        repos = len({t.repo for t in targets})
        if preview is not None:
            run_refetch_preview(data_dir, targets, preview, examples, seed)
            return
        if not apply:
            console.print(
                f"Would refetch {len(targets)} logs from {repos} repos."
                " Dry run: nothing downloaded. Rerun with --apply to refetch."
            )
            return

        def report_expired(target: RefetchTarget) -> None:
            console.print(
                f"  [yellow]expired[/yellow] {target.repo} job {target.job_id}:"
                " kept old log, marked over_redacted"
            )

        console.print(f"Refetching {len(targets)} logs from {repos} repos...")
        try:
            with open_github() as client:
                result = refetch_logs(client, conn, data_dir, targets, on_expired=report_expired)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow] Finished logs are saved; rerun to resume.")
            raise typer.Exit(130) from None
    finally:
        conn.close()
    console.print(
        f"Refetched {result.refetched}, expired {result.expired}, failed {len(result.failed)}."
    )
    if result.failed:
        console.print("[yellow]Failed logs are unchanged; rerun to retry them.[/yellow]")


def run_refetch_preview(
    data_dir: Path, targets: list[RefetchTarget], sample: int, examples: int, seed: int | None
) -> None:
    rng = random.Random(seed)  # noqa: S311 - sampling only, not security
    with open_github() as client:
        result = preview_refetch(client, data_dir, targets, sample, rng)
    console.print(
        f"Preview: downloaded {len(result.logs)} of {result.targets} target logs"
        f" (expired {len(result.expired)}, failed {len(result.failed)})."
    )
    console.print("[dim]Nothing written to disk or the DB.[/dim]")
    if result.logs:
        before = ", ".join(f"{rule} {n}" for rule, n in result.stored_counts.most_common())
        console.print(f"Stored versions of these logs: {before or 'no masks'}.")
    report = build_redaction_report(result.logs, rng, examples_per_rule=examples)
    print_redaction_report(report, empty="The current rules mask nothing in these logs.")


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


label_app = typer.Typer(
    help="Label failures by hand (run without a subcommand to start a session).",
    invoke_without_command=True,
)
app.add_typer(label_app, name="label")


@label_app.callback()
def label(
    ctx: typer.Context,
    n: Annotated[int, typer.Option("--n", help="How many failures to label this session.")] = 50,
    seed: Annotated[int | None, typer.Option(help="Random seed for a reproducible pick.")] = None,
    max_per_run: Annotated[
        int,
        typer.Option(
            min=0,
            help="Max jobs to label per workflow run, counting earlier labels (0 = no limit).",
        ),
    ] = 1,
    include_over_redacted: Annotated[
        bool,
        typer.Option(
            "--include-over-redacted",
            help="Also offer logs marked over_redacted (old masks hid context; log expired).",
        ),
    ] = False,
    split: Annotated[
        str,
        typer.Option(help="Split recorded on this session's new labels: dev, test1 or test2."),
    ] = TEST2_SPLIT,
) -> None:
    """Label unlabeled failures by hand; answers are appended to labels.jsonl."""
    if ctx.invoked_subcommand is not None:
        return
    if split not in SPLITS:
        console.print(f"[red]--split must be one of {', '.join(SPLITS)}.[/red]")
        raise typer.Exit(2)
    data_dir = get_settings().data_dir
    labels_path = data_dir / LABELS_FILENAME

    def warn_bad_line(lineno: int) -> None:
        console.print(f"[yellow]ignoring unreadable line {lineno} in {labels_path}[/yellow]")

    existing = load_labels(labels_path, on_bad_line=warn_bad_line)
    conn = open_db()
    try:
        labeled_ids = {lbl.job_id for lbl in existing}
        candidates = load_candidates(conn, labeled_ids, include_over_redacted)
        hidden = 0 if include_over_redacted else count_unlabeled_over_redacted(conn, labeled_ids)
    finally:
        conn.close()
    if hidden:
        console.print(
            f"[dim]Skipping {hidden} over-redacted logs"
            " (--include-over-redacted to label them).[/dim]"
        )
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
        f" New labels go to the {split} split."
    )
    result = run_session(
        console,
        picks,
        data_dir,
        labels_path,
        getch=typer.getchar,
        ask=console.input,
        clock=lambda: datetime.now(UTC),
        split=split,
    )
    status = "Stopped" if result.quit else "Done"
    console.print(
        f"{status}: {result.labeled} labeled, {result.skipped} skipped "
        f"({len(existing) + result.labeled} total in {labels_path}). Run again to resume."
    )


@label_app.command("edit")
def label_edit(
    job_id: Annotated[int | None, typer.Argument(help="Job id of the label to edit.")] = None,
    last: Annotated[
        bool, typer.Option("--last", help="Edit the most recently written label.")
    ] = False,
) -> None:
    """Change a label's category, root cause, evidence or confidence (Enter keeps each)."""
    if (job_id is None) == (not last):
        console.print("[red]Give a job id or --last (not both).[/red]")
        raise typer.Exit(2)
    data_dir = get_settings().data_dir
    labels_path = data_dir / LABELS_FILENAME
    try:
        stored = find_label(labels_path, None if last else job_id)
    except LabelNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    try:
        outcome = edit_stored_label(labels_path, data_dir, stored)
    except QuitRequestedError:
        console.print("[yellow]Quit: label unchanged.[/yellow]")
        raise typer.Exit(0) from None
    if outcome == "missing_log":
        raise typer.Exit(1)


EditOutcome = Literal["saved", "unchanged", "missing_log"]


def edit_stored_label(
    labels_path: Path, data_dir: Path, stored: StoredLabel, index: int = 1, total: int = 1
) -> EditOutcome:
    """Show the label's log and run the edit prompts; save only if something changed.

    Shows the log, the job header and the current label: never a model prediction or a
    hint. Raises QuitRequestedError on quit, leaving this label unchanged.
    """
    current = stored.label
    log_file = data_dir / current.log_path
    if not log_file.exists():
        console.print(f"[red]log missing:[/red] {log_file}; cannot show or check evidence lines.")
        return "missing_log"
    conn = open_db()
    try:
        candidate = load_candidate(conn, current.job_id)
    finally:
        conn.close()
    if candidate is None:  # pruned from the DB; the label and log still exist
        candidate = Candidate(
            job_id=current.job_id,
            repo=current.repo,
            run_id=current.run_id,
            workflow_name="?",
            workflow_path="?",
            event="?",
            head_branch=None,
            failed_steps=[],
            runner_os=None,
            log_path=current.log_path,
        )

    text = read_log(log_file)
    full = build_view(text)
    views = ViewToggle(console, full, build_focus_view(text, candidate.failed_steps))
    render_header(console, candidate, index, total)
    views.render()
    console.print(
        f"[bold]Editing label for job {current.job_id}[/bold]"
        f" (labeled {current.labeled_at}). Enter keeps the current value."
    )
    changes = edit_one(console, typer.getchar, console.input, current, full.total_lines, views)

    changed = [k for k, v in changes.items() if stored.record.get(k) != v]
    if not changed:
        console.print("No changes; label unchanged.")
        return "unchanged"
    record = {**stored.record, **changes, "edited_at": datetime.now(UTC).isoformat()}
    replace_label_line(labels_path, stored.line_index, record)
    console.print(f"Saved: changed {', '.join(changed)} for job {current.job_id}.")
    return "saved"


@label_app.command("review")
def label_review(
    split: Annotated[
        str, typer.Option(help="Split to review. Only dev: test labels are never re-checked.")
    ] = DEV_SPLIT,
    category: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated categories to review, e.g. other,ci_config (default: all)."
        ),
    ] = None,
) -> None:
    """Re-check dev labels one by one with the `label edit` flow (Enter keeps each value).

    Shows only the log and the current label, never a model prediction, so the
    review is decided from the log and the guideline alone.
    """
    if split != DEV_SPLIT:
        console.print(
            f"[red]Refusing to review the {split} split: only dev labels may be re-checked"
            " (test labels are reported as-is and never tuned on).[/red]"
        )
        raise typer.Exit(2)
    wanted = set(CATEGORIES)
    if category is not None:
        wanted = {c.strip() for c in category.split(",") if c.strip()}
        unknown = sorted(wanted - set(CATEGORIES))
        if unknown or not wanted:
            console.print(
                f"[red]Unknown categories {unknown}; choose from {', '.join(CATEGORIES)}.[/red]"
            )
            raise typer.Exit(2)
    data_dir = get_settings().data_dir
    labels_path = data_dir / LABELS_FILENAME
    picks = [
        label.job_id
        for label in newest_labels(load_labels(labels_path)).values()
        if label.split == DEV_SPLIT and label.category in wanted
    ]
    if not picks:
        console.print("No dev labels in those categories.")
        return
    console.print(
        f"Reviewing {len(picks)} dev labels. Decide from the log and the guideline only,"
        " and record the reason in the root cause. q stops; reviewed labels are kept."
    )
    counts = {"saved": 0, "unchanged": 0, "missing_log": 0}
    for index, job_id in enumerate(picks, start=1):
        # Looked up again each time, so the line index is current.
        stored = find_label(labels_path, job_id)
        try:
            counts[edit_stored_label(labels_path, data_dir, stored, index, len(picks))] += 1
        except QuitRequestedError:
            console.print(
                f"[yellow]Stopped at {index} of {len(picks)}: job {job_id} unchanged.[/yellow]"
            )
            break
    console.print(
        f"Reviewed: {counts['saved']} changed, {counts['unchanged']} kept,"
        f" {counts['missing_log']} skipped (log missing)."
    )
