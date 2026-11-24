"""Resolve what `cidoc analyze` was given (a log path or a stored job id) into log text.

Nothing leaves the machine unredacted:
- A stored log (by job id, or a path under <data_dir>/logs) was redacted by the miner
  when it was saved. It is used as is, so line numbers match the labels exactly.
- Any other file may hold live secrets. It is redacted with the current rules first.
  The private-key rule can join lines, so line numbers then refer to the redacted text.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cidoc.labeling.labels import LABELS_FILENAME, LabelNotFoundError, find_label
from cidoc.labeling.session import load_candidate
from cidoc.logview import read_log, split_lines
from cidoc.redact import redact

LOGS_DIRNAME = "logs"
_GZIP_MAGIC = b"\x1f\x8b"


class InputError(ValueError):
    """The target is neither a readable log file nor a known job id."""


@dataclass(frozen=True)
class AnalysisInput:
    text: str
    source: str
    job_id: int | None = None
    failed_steps: list[str] = field(default_factory=list)
    redactions: int = 0
    """Values masked now (always 0 for stored logs, which were redacted when mined)."""
    renumbered: bool = False
    """Redaction changed the line count, so line numbers differ from the original file."""


def read_log_file(path: Path) -> str:
    """A gzip or plain-text log, keeping bare CRs so line numbers stay put."""
    with path.open("rb") as fh:
        gzipped = fh.read(2) == _GZIP_MAGIC
    if gzipped:
        return read_log(path)
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _is_stored(path: Path, data_dir: Path) -> bool:
    return path.resolve().is_relative_to((data_dir / LOGS_DIRNAME).resolve())


def from_path(path: Path, data_dir: Path) -> AnalysisInput:
    try:
        text = read_log_file(path)
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from None
    if _is_stored(path, data_dir):
        return AnalysisInput(text=text, source=str(path))
    redaction = redact(text)
    return AnalysisInput(
        text=redaction.text,
        source=str(path),
        redactions=redaction.total,
        renumbered=len(split_lines(redaction.text)) != len(split_lines(text)),
    )


def from_job_id(
    job_id: int, data_dir: Path, open_db: Callable[[], sqlite3.Connection]
) -> AnalysisInput:
    conn = open_db()
    try:
        candidate = load_candidate(conn, job_id)
    finally:
        conn.close()
    if candidate is not None:
        log_path, failed_steps = candidate.log_path, candidate.failed_steps
    else:  # pruned from the DB, but a label may still point at its log
        try:
            log_path = find_label(data_dir / LABELS_FILENAME, job_id).label.log_path
        except LabelNotFoundError:
            raise InputError(f"job {job_id} is neither in the database nor labeled") from None
        failed_steps = []
    path = data_dir / log_path
    if not path.exists():
        raise InputError(f"log for job {job_id} is missing: {path}")
    return AnalysisInput(
        text=read_log(path), source=str(path), job_id=job_id, failed_steps=failed_steps
    )


def resolve_input(
    target: str, data_dir: Path, open_db: Callable[[], sqlite3.Connection]
) -> AnalysisInput:
    """An existing file wins; otherwise a number is a job id."""
    path = Path(target)
    if path.is_file():
        return from_path(path, data_dir)
    if target.isdigit():
        return from_job_id(int(target), data_dir, open_db)
    raise InputError(f"{target} is neither a log file nor a job id")
