"""Interactive labeling loop: show a failure, take single-key answers, append to labels.jsonl."""

import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from cidoc.labeling.display import (
    FocusView,
    LogView,
    build_focus_view,
    build_view,
    read_log,
    render_focus_view,
    render_view,
)
from cidoc.labeling.hints import Hint, hint_for_log
from cidoc.labeling.labels import (
    CATEGORY_HINTS,
    CATEGORY_KEYS,
    CONFIDENCE_LEVELS,
    Label,
    append_label,
    parse_evidence,
)

SKIP_KEY = "s"
TOGGLE_VIEW_KEY = "f"
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


_CANDIDATE_COLUMNS = (
    "job_id, repo, run_id, workflow_name, workflow_path, event, head_branch,"
    " failed_steps, runner_os, log_path"
)


def _candidate(row: sqlite3.Row) -> Candidate:
    return Candidate(
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


def load_candidate(conn: sqlite3.Connection, job_id: int) -> Candidate | None:
    row = conn.execute(
        f"SELECT {_CANDIDATE_COLUMNS} FROM failures WHERE job_id = ?",  # noqa: S608 - constant
        (job_id,),
    ).fetchone()
    return _candidate(row) if row else None


def load_candidates(
    conn: sqlite3.Connection, exclude: set[int], include_over_redacted: bool = False
) -> list[Candidate]:
    """Unlabeled failures, minus `exclude`.

    Logs marked over_redacted lost context to old redaction rules and could not be
    refetched; they are left out unless `include_over_redacted` is set.
    """
    rows = conn.execute(
        f"SELECT {_CANDIDATE_COLUMNS} FROM failures"  # noqa: S608 - constant columns
        " WHERE over_redacted = 0 OR ? ORDER BY job_id",
        (include_over_redacted,),
    )
    return [_candidate(row) for row in rows if row["job_id"] not in exclude]


def count_unlabeled_over_redacted(conn: sqlite3.Connection, exclude: set[int]) -> int:
    """How many logs load_candidates leaves out by default."""
    rows = conn.execute("SELECT job_id FROM failures WHERE over_redacted = 1")
    return sum(row[0] not in exclude for row in rows)


def cap_per_run(
    candidates: Sequence[Candidate],
    already_labeled_runs: Iterable[tuple[str, int]],
    max_per_run: int,
    rng: random.Random,
) -> list[Candidate]:
    """Keep at most `max_per_run` jobs per (repo, run_id), counting earlier sessions' labels.

    Failed jobs in one run are often matrix legs with the same error, so labeling more
    than one mostly repeats work and skews the label set toward wide matrices. Which
    jobs are kept is random. Input order is preserved.
    """
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    taken = Counter(already_labeled_runs)
    keep: set[int] = set()
    for c in shuffled:
        key = (c.repo, c.run_id)
        if taken[key] < max_per_run:
            taken[key] += 1
            keep.add(c.job_id)
    return [c for c in candidates if c.job_id in keep]


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


def render_menu(console: Console, can_toggle: bool = False) -> None:
    entries = []
    for key, category in CATEGORY_KEYS.items():
        hint = CATEGORY_HINTS.get(category)
        entries.append(f"[cyan]{key}[/cyan] {category}" + (f" [dim]({hint})[/dim]" if hint else ""))
    if can_toggle:
        entries.append(f"[cyan]{TOGGLE_VIEW_KEY}[/cyan] toggle failed-step focus / full view")
    entries += [f"[cyan]{SKIP_KEY}[/cyan] skip", "[cyan]q[/cyan] quit"]
    console.print("\n".join(entries))


class ViewToggle:
    """Shows the failed-step focus view first; `toggle` flips to the full view and back.

    Falls back to the full view, with no toggle, when no failed step can be located.
    """

    def __init__(self, console: Console, full: LogView, focus: FocusView) -> None:
        self._console = console
        self._full = full
        self._focus = focus
        self.can_toggle = not focus.empty
        self.showing_focus = self.can_toggle

    def render(self) -> None:
        if self.showing_focus:
            render_focus_view(self._console, self._focus)
            return
        if not self.can_toggle:
            self._console.print(
                Text("no failed step or ##[error] found; showing the full view", style="dim")
            )
        render_view(self._console, self._full)

    def toggle(self) -> None:
        self.showing_focus = not self.showing_focus
        self.render()


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


def render_hint(console: Console, hint: Hint) -> None:
    # Text, not markup: the matched snippet is log content.
    line = Text("hint: ", style="bold yellow")
    line.append(f"{hint.category} (matched: {hint.matched!r} on L{hint.line_no})")
    console.print(line, highlight=False)


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
    views: ViewToggle | None = None,
    hint: Hint | None = None,
) -> Label | None:
    """Collect one label, or return None if skipped. Raises QuitRequestedError on quit.

    The hint is only displayed: the category is always typed by the labeler.
    """
    can_toggle = views is not None and views.can_toggle
    keys = [*CATEGORY_KEYS, SKIP_KEY, *([TOGGLE_VIEW_KEY] if can_toggle else [])]
    while True:
        render_menu(console, can_toggle)
        if hint is not None:
            render_hint(console, hint)
        key = read_key(console, getch, keys, "category>")
        if key != TOGGLE_VIEW_KEY or views is None:
            break
        views.toggle()
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
    category = CATEGORY_KEYS[key]
    return Label(
        job_id=c.job_id,
        repo=c.repo,
        run_id=c.run_id,
        category=category,
        root_cause=root_cause,
        evidence_lines=evidence,
        confidence=int(confidence),
        log_path=c.log_path,
        labeled_at=now.isoformat(),
        hint=hint.category if hint else None,
        hint_accepted=(category == hint.category) if hint else None,
    )


KEEP_KEYS = frozenset({"\r", "\n"})


def keep_or[V](parse: Callable[[str], V], current: V) -> Callable[[str], V]:
    """A parser where an empty answer means 'keep the current value'."""
    return lambda text: current if not text.strip() else parse(text)


def edit_one(
    console: Console,
    getch: Getch,
    ask: Ask,
    current: Label,
    total_lines: int,
    views: ViewToggle | None = None,
) -> dict[str, object]:
    """Ask for each editable field with the current value pre-filled; Enter keeps it.

    Returns only the editable fields (category, root_cause, evidence_lines, confidence).
    Raises QuitRequestedError on quit, in which case nothing is changed.
    """
    can_toggle = views is not None and views.can_toggle
    keys = [*CATEGORY_KEYS, *KEEP_KEYS, *([TOGGLE_VIEW_KEY] if can_toggle else [])]
    while True:
        render_menu(console, can_toggle)
        key = read_key(console, getch, keys, f"category [{current.category}]>")
        if key != TOGGLE_VIEW_KEY or views is None:
            break
        views.toggle()
    category = current.category if key in KEEP_KEYS else CATEGORY_KEYS[key]

    root_cause = ask_until_valid(
        console,
        ask,
        f"root cause [{current.root_cause}]> ",
        keep_or(parse_root_cause, current.root_cause),
    )
    shown = ",".join(map(str, current.evidence_lines))
    evidence = ask_until_valid(
        console,
        ask,
        f"evidence line numbers [{shown}] (1-{total_lines})> ",
        keep_or(lambda text: parse_evidence(text, total_lines), current.evidence_lines),
    )
    confidence = read_key(
        console,
        getch,
        [*(str(n) for n in CONFIDENCE_LEVELS), *KEEP_KEYS],
        f"confidence 1-3 [{current.confidence}]>",
    )
    return {
        "category": category,
        "root_cause": root_cause,
        "evidence_lines": evidence,
        "confidence": current.confidence if confidence in KEEP_KEYS else int(confidence),
    }


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
        text = read_log(log_file)
        view = build_view(text)
        focus = build_focus_view(text, c.failed_steps)
        views = ViewToggle(console, view, focus)
        hint = hint_for_log(text, focus)
        render_header(console, c, index, len(picks))
        views.render()
        try:
            label = label_one(console, getch, ask, c, view.total_lines, clock(), views, hint)
        except QuitRequestedError:
            result.quit = True
            break
        if label is None:
            result.skipped += 1
            continue
        append_label(labels_path, label)
        result.labeled += 1
    return result
