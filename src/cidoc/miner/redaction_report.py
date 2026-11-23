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
from dataclasses import dataclass
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


def collect_redaction_report(
    conn: sqlite3.Connection,
    data_dir: Path,
    rng: random.Random,
    examples_per_rule: int = EXAMPLES_PER_RULE,
    context: int = CONTEXT_CHARS,
) -> RedactionReport:
    counts: Counter[str] = Counter()
    logs_per_rule: Counter[str] = Counter()
    found: defaultdict[str, list[Example]] = defaultdict(list)
    from_markers = 0
    missing: list[int] = []

    rows = conn.execute(
        "SELECT job_id, repo, log_path, redaction_counts FROM failures"
        " WHERE redactions > 0 ORDER BY job_id"
    ).fetchall()
    for row in rows:
        path = data_dir / row["log_path"]
        if not path.exists():
            missing.append(row["job_id"])
            continue
        text = read_log(path)
        # Examples come from cleaned lines (no ANSI codes or timestamps); line numbers
        # still match the stored log because cleaning keeps one line per line.
        for line_no, line in enumerate(clean_log(text), start=1):
            for match in MASK_RE.finditer(line):
                snippet = snippet_around(line, match.start(), match.end(), context)
                found[match.group(1)].append(Example(row["job_id"], row["repo"], line_no, snippet))
        stored: dict[str, int] = json.loads(row["redaction_counts"])
        if not stored:
            # Count on the raw text: cleaning drops text before a bare CR, markers included.
            stored = dict(Counter(MASK_RE.findall(text)))
            from_markers += 1
        counts.update(stored)
        logs_per_rule.update(stored.keys())

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
        logs_with_redactions=len(rows) - len(missing),
        logs_counted_from_markers=from_markers,
        logs_missing=missing,
    )
