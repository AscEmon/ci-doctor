"""Audit what the redactor masked: counts per rule and masked examples in context.

Examples are cut from the stored logs, which were redacted before they were written,
so the matched values are already gone from disk. Each snippet is passed through the
redactor once more before it is returned, in case the rules are stricter now than
when the log was mined.
"""

import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from cidoc.labeling.display import clean_log, read_log
from cidoc.redact import MASK_RE, RULE_NAMES, redact

EXAMPLES_PER_RULE = 10
CONTEXT_CHARS = 40
ELLIPSIS = "…"


@dataclass(frozen=True)
class Example:
    job_id: int
    repo: str
    line_no: int
    snippet: str


@dataclass(frozen=True)
class RuleReport:
    rule: str
    redactions: int
    logs: int
    examples: list[Example]


@dataclass(frozen=True)
class RedactionReport:
    rules: list[RuleReport]
    logs_with_redactions: int
    logs_counted_from_markers: int
    """Logs mined before per-rule counts were stored; counted from their markers instead."""
    logs_missing: list[int]
    """job_ids whose log file is gone, so they contribute neither counts nor examples."""


def snippet_around(line: str, start: int, end: int, context: int) -> str:
    """`line[start:end]` (a mask marker) with up to `context` chars either side, re-redacted."""
    left = max(0, start - context)
    right = min(len(line), end + context)
    text = line[left:right]
    text = (ELLIPSIS if left > 0 else "") + text + (ELLIPSIS if right < len(line) else "")
    return redact(text).text


@dataclass(frozen=True)
class LogText:
    """One redacted log to report on."""

    job_id: int
    repo: str
    text: str
    counts: dict[str, int] | None
    """Per-rule counts if known ({} for none); None to count the masks in `text` instead."""


def build_redaction_report(
    logs: Iterable[LogText],
    rng: random.Random,
    examples_per_rule: int = EXAMPLES_PER_RULE,
    context: int = CONTEXT_CHARS,
) -> RedactionReport:
    """Per-rule counts and sampled masked examples for already-redacted log texts."""
    counts: Counter[str] = Counter()
    logs_per_rule: Counter[str] = Counter()
    found: defaultdict[str, list[Example]] = defaultdict(list)
    from_markers = 0
    with_redactions = 0

    for log in logs:
        # Examples come from cleaned lines (no ANSI codes or timestamps); line numbers
        # still match the stored log because cleaning keeps one line per line.
        for line_no, line in enumerate(clean_log(log.text), start=1):
            for match in MASK_RE.finditer(line):
                snippet = snippet_around(line, match.start(), match.end(), context)
                found[match.group(1)].append(Example(log.job_id, log.repo, line_no, snippet))
        log_counts = log.counts
        if log_counts is None:
            # Count on the raw text: cleaning drops text before a bare CR, markers included.
            log_counts = dict(Counter(MASK_RE.findall(log.text)))
            from_markers += 1
        if any(log_counts.values()):
            with_redactions += 1
        counts.update(log_counts)
        logs_per_rule.update(k for k, v in log_counts.items() if v)

    # Known rules first in redactor order, then anything unexpected found in old logs.
    ordered = [r for r in RULE_NAMES if r in counts] + sorted(set(counts) - set(RULE_NAMES))
    rules = [
        RuleReport(
            rule=rule,
            redactions=counts[rule],
            logs=logs_per_rule[rule],
            examples=rng.sample(found[rule], min(examples_per_rule, len(found[rule]))),
        )
        for rule in ordered
    ]
    return RedactionReport(
        rules=rules,
        logs_with_redactions=with_redactions,
        logs_counted_from_markers=from_markers,
        logs_missing=[],
    )


def collect_redaction_report(
    conn: sqlite3.Connection,
    data_dir: Path,
    rng: random.Random,
    examples_per_rule: int = EXAMPLES_PER_RULE,
    context: int = CONTEXT_CHARS,
) -> RedactionReport:
    """Report on every stored log with redactions."""
    missing: list[int] = []
    rows = conn.execute(
        "SELECT job_id, repo, log_path, redaction_counts FROM failures"
        " WHERE redactions > 0 ORDER BY job_id"
    ).fetchall()

    def stored_logs() -> Iterator[LogText]:
        for row in rows:
            path = data_dir / row["log_path"]
            if not path.exists():
                missing.append(row["job_id"])
                continue
            # '{}' means the row predates per-rule counts: count the stored masks.
            stored: dict[str, int] = json.loads(row["redaction_counts"])
            yield LogText(row["job_id"], row["repo"], read_log(path), stored or None)

    report = build_redaction_report(stored_logs(), rng, examples_per_rule, context)
    return replace(report, logs_missing=missing)
