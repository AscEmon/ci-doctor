"""Interactive labeling loop: show a failure, take single-key answers, append to labels.jsonl."""

import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from cidoc.labeling.display import build_view, read_log, render_view
from cidoc.labeling.labels import (
    CATEGORY_HINTS,
    CATEGORY_KEYS,
    CONFIDENCE_LEVELS,
    Label,
    append_label,
    parse_evidence,
)

SKIP_KEY = "s"
QUIT_KEYS = frozenset({"q", "\x03", "\x04"})  # q, Ctrl-C, Ctrl-D

Getch = Callable[[], str]
Ask = Callable[[str], str]


@dataclass(frozen=True)
class Candidate:
    job_id: int
    repo: str
    run_id: int
    workflow_name: str
    workflow_path: str
    event: str
    head_branch: str | None
    failed_steps: list[str]
    runner_os: str | None
    log_path: str


@dataclass
class SessionResult:
    labeled: int = 0
    skipped: int = 0
    quit: bool = False


class QuitRequestedError(Exception):
    """The labeler asked to stop; nothing partial is written."""


def load_candidates(conn: sqlite3.Connection, exclude: set[int]) -> list[Candidate]:
    rows = conn.execute(
        "SELECT job_id, repo, run_id, workflow_name, workflow_path, event, head_branch,"
        " failed_steps, runner_os, log_path FROM failures ORDER BY job_id"
    )
    return [
        Candidate(
            job_id=row["job_id"],
            repo=row["repo"],
            run_id=row["run_id"],
            workflow_name=row["workflow_name"],
            workflow_path=row["workflow_path"],
            event=row["event"],
            head_branch=row["head_branch"],
            failed_steps=json.loads(row["failed_steps"]),
            runner_os=row["runner_os"],
            log_path=row["log_path"],
        )
        for row in rows
        if row["job_id"] not in exclude
    ]


def pick_stratified(
    candidates: Iterable[Candidate],
    already_labeled_repos: Iterable[str],
    n: int,
    rng: random.Random,
) -> list[Candidate]:
    """Pick up to n, always from the repo with the fewest labels so far (ties broken randomly).

    Counting earlier sessions' labels keeps coverage balanced across sessions, not just within one.
    """
    pools: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        pools[c.repo].append(c)
    for pool in pools.values():
        rng.shuffle(pool)
    counts = Counter(already_labeled_repos)
    tiebreak = {repo: rng.random() for repo in pools}
    picked: list[Candidate] = []
    while len(picked) < n and pools:
        repo = min(pools, key=lambda r: (counts[r], tiebreak[r]))
        picked.append(pools[repo].pop())
        counts[repo] += 1
        if not pools[repo]:
            del pools[repo]
    return picked


def render_header(console: Console, c: Candidate, index: int, total: int) -> None:
    console.rule(f"[bold]{index}/{total}[/bold]  job {c.job_id}")
    rows = (
        ("repo", c.repo),
        ("workflow", f"{c.workflow_name} ({c.workflow_path})"),
        ("event", f"{c.event} on {c.head_branch or '?'}  runner: {c.runner_os or '?'}"),
        ("failed steps", ", ".join(c.failed_steps) or "(none recorded)"),
        ("url", f"https://github.com/{c.repo}/actions/runs/{c.run_id}/job/{c.job_id}"),
    )
    for name, value in rows:
        line = Text(f"{name:>12}: ", style="bold")
        line.append(value)
        console.print(line, highlight=False)


def render_menu(console: Console) -> None:
    entries = []
    for key, category in CATEGORY_KEYS.items():
        hint = CATEGORY_HINTS.get(category)
        entries.append(f"[cyan]{key}[/cyan] {category}" + (f" [dim]({hint})[/dim]" if hint else ""))
    entries += [f"[cyan]{SKIP_KEY}[/cyan] skip", "[cyan]q[/cyan] quit"]
    console.print("\n".join(entries))


def read_key(console: Console, getch: Getch, valid: Iterable[str], prompt: str) -> str:
    allowed = set(valid)
    console.print(prompt, end=" ")
    while True:
        key = getch().lower()
        if key in QUIT_KEYS:
            console.print()
            raise QuitRequestedError
        if key in allowed:
            console.print(key)
            return key


def ask_until_valid[T](console: Console, ask: Ask, prompt: str, parse: Callable[[str], T]) -> T:
    while True:
        try:
            answer = ask(prompt)
        except (KeyboardInterrupt, EOFError):
            raise QuitRequestedError from None
        try:
            return parse(answer)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")


def parse_root_cause(text: str) -> str:
    one_line = " ".join(text.split())
    if not one_line:
        raise ValueError("root cause cannot be empty")
    return one_line


def label_one(
    console: Console,
    getch: Getch,
    ask: Ask,
    c: Candidate,
    total_lines: int,
    now: datetime,
) -> Label | None:
    """Collect one label, or return None if skipped. Raises QuitRequestedError on quit."""
    render_menu(console)
    key = read_key(console, getch, [*CATEGORY_KEYS, SKIP_KEY], "category>")
    if key == SKIP_KEY:
        return None
    root_cause = ask_until_valid(console, ask, "root cause (one line)> ", parse_root_cause)
    evidence = ask_until_valid(
        console,
        ask,
        f"evidence line numbers (comma-separated, 1-{total_lines})> ",
        lambda text: parse_evidence(text, total_lines),
    )
    confidence = read_key(console, getch, [str(n) for n in CONFIDENCE_LEVELS], "confidence 1-3>")
    return Label(
        job_id=c.job_id,
        repo=c.repo,
        run_id=c.run_id,
        category=CATEGORY_KEYS[key],
        root_cause=root_cause,
        evidence_lines=evidence,
        confidence=int(confidence),
        log_path=c.log_path,
        labeled_at=now.isoformat(),
    )


def run_session(
    console: Console,
    picks: list[Candidate],
    data_dir: Path,
    labels_path: Path,
    getch: Getch,
    ask: Ask,
    clock: Callable[[], datetime],
) -> SessionResult:
    result = SessionResult()
    for index, c in enumerate(picks, start=1):
        log_file = data_dir / c.log_path
        if not log_file.exists():
            console.print(f"[yellow]log missing, skipping[/yellow] {c.log_path}")
            result.skipped += 1
            continue
        view = build_view(read_log(log_file))
        render_header(console, c, index, len(picks))
        render_view(console, view)
        try:
            label = label_one(console, getch, ask, c, view.total_lines, clock())
        except QuitRequestedError:
            result.quit = True
            break
        if label is None:
            result.skipped += 1
            continue
        append_label(labels_path, label)
        result.labeled += 1
    return result
