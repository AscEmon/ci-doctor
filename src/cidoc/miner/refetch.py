"""Re-download logs that older redaction rules over-masked, and redact them again.

A log is a target when it was redacted with an older REDACTOR_VERSION and its stored
file holds a mask from a generic rule (the retired keyword_secret, or the current
assignment_secret), since those are the rules that change. Refetched logs are stamped
with the current version, so running this again does not download them again, even
when the new rules still (rightly) mask something in them.

preview_refetch downloads a sample of targets and redacts them in memory only, to
check what new rules would do before rewriting anything.
"""

import json
import logging
import random
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cidoc.github_client import GitHubClient
from cidoc.labeling.display import read_log
from cidoc.miner.redaction_report import LogText
from cidoc.miner.runs import download_job_log, redact_log, store_log
from cidoc.redact import MASK_RE, REDACTOR_VERSION

logger = logging.getLogger(__name__)

# Masks from generic (heuristic) rules: the ones a rule change can make wrong.
GENERIC_MASKS = ("[REDACTED:keyword_secret]", "[REDACTED:assignment_secret]")


@dataclass(frozen=True)
class RefetchTarget:
    job_id: int
    repo: str
    log_path: str


@dataclass
class RefetchResult:
    refetched: int = 0
    expired: int = 0
    """GitHub no longer has the log; the old one is kept and marked over_redacted."""
    failed: list[tuple[int, str]] = field(default_factory=list)
    """(job_id, error type) for downloads that failed for another reason."""


def find_refetch_targets(conn: sqlite3.Connection, data_dir: Path) -> list[RefetchTarget]:
    """Logs from older redactor versions whose stored file contains a generic-rule mask.

    Logs already marked over_redacted are skipped: GitHub said they are gone.
    """
    rows = conn.execute(
        "SELECT job_id, repo, log_path FROM failures"
        " WHERE redactions > 0 AND over_redacted = 0 AND redactor_version < ?"
        " ORDER BY repo, job_id",
        (REDACTOR_VERSION,),
    ).fetchall()
    targets: list[RefetchTarget] = []
    for row in rows:
        path = data_dir / row["log_path"]
        if path.exists() and _has_generic_mask(read_log(path)):
            targets.append(RefetchTarget(row["job_id"], row["repo"], row["log_path"]))
    return targets


def _has_generic_mask(text: str) -> bool:
    return any(mask in text for mask in GENERIC_MASKS)


def count_over_redacted(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM failures WHERE over_redacted = 1").fetchone()
    count: int = row[0]
    return count


def _save_refetched(
    conn: sqlite3.Connection, job_id: int, log_bytes: int, counts: dict[str, int]
) -> None:
    with conn:
        conn.execute(
            "UPDATE failures SET log_bytes = ?, redactions = ?, redaction_counts = ?,"
            " redactor_version = ?, over_redacted = 0 WHERE job_id = ?",
            (
                log_bytes,
                sum(counts.values()),
                json.dumps(counts, sort_keys=True),
                REDACTOR_VERSION,
                job_id,
            ),
        )


def _mark_over_redacted(conn: sqlite3.Connection, job_id: int) -> None:
    with conn:
        conn.execute("UPDATE failures SET over_redacted = 1 WHERE job_id = ?", (job_id,))


def refetch_logs(
    client: GitHubClient,
    conn: sqlite3.Connection,
    data_dir: Path,
    targets: list[RefetchTarget],
    on_expired: Callable[[RefetchTarget], None] | None = None,
) -> RefetchResult:
    """Download, redact and overwrite each target's log, updating its row.

    The new log replaces the old one atomically (store_log writes a temp file and renames).
    An unexpected error skips that log and leaves it untouched for the next run.
    KeyboardInterrupt still stops; finished logs are already saved.
    """
    result = RefetchResult()
    for target in targets:
        try:
            raw = download_job_log(client, target.repo, target.job_id)
            if raw is None:
                _mark_over_redacted(conn, target.job_id)
                result.expired += 1
                if on_expired:
                    on_expired(target)
                continue
            log_bytes, counts = store_log(data_dir, Path(target.log_path), raw)
            _save_refetched(conn, target.job_id, log_bytes, counts)
            result.refetched += 1
        except Exception as exc:
            logger.warning(
                "refetch failed: %s job %s: %s", target.repo, target.job_id, type(exc).__name__
            )
            result.failed.append((target.job_id, type(exc).__name__))
    return result


@dataclass
class PreviewResult:
    targets: int
    """How many targets there were to sample from."""
    logs: list[LogText] = field(default_factory=list)
    """Sampled logs as the current rules redact them, in memory only."""
    stored_counts: Counter[str] = field(default_factory=Counter)
    """Masks per rule in the stored versions of the downloaded logs, for comparison."""
    expired: list[RefetchTarget] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)


def preview_refetch(
    client: GitHubClient,
    data_dir: Path,
    targets: list[RefetchTarget],
    sample: int,
    rng: random.Random,
) -> PreviewResult:
    """Download up to `sample` random targets and redact them in memory.

    Reads the stored logs only to count their masks; writes nothing to disk or the DB.
    """
    result = PreviewResult(targets=len(targets))
    for target in rng.sample(targets, min(sample, len(targets))):
        try:
            raw = download_job_log(client, target.repo, target.job_id)
        except Exception as exc:
            logger.warning(
                "preview download failed: %s job %s: %s",
                target.repo,
                target.job_id,
                type(exc).__name__,
            )
            result.failed.append((target.job_id, type(exc).__name__))
            continue
        if raw is None:
            result.expired.append(target)
            continue
        redacted = redact_log(raw)
        result.logs.append(
            LogText(target.job_id, target.repo, redacted.text, dict(redacted.counts))
        )
        stored = read_log(data_dir / target.log_path)
        result.stored_counts.update(MASK_RE.findall(stored))
    return result
