"""Failure taxonomy and the labels.jsonl store.

New labels are appended. `cidoc label edit` rewrites a single line in place (atomically),
leaving every other line untouched.
"""

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

LABELS_FILENAME = "labels.jsonl"

# Single keypress -> category. Order follows the taxonomy in the project spec.
CATEGORY_KEYS: dict[str, str] = {
    "1": "dependency_resolution",
    "2": "gradle_agp_config",
    "3": "native_compile",
    "4": "dart_compile",
    "5": "analyzer_lint",
    "6": "test_failure",
    "7": "signing_secrets",
    "8": "toolchain_version",
    "9": "ios_build",
    "0": "infra_flaky",
    "t": "timeout_resource",
    "c": "ci_config",  # added in the Phase 0 wrap-up, after the first 44 labels
    "o": "other",
}

# Every category, in menu order. The one list other code (e.g. an analyzer's output
# schema) should take its enum from. Categories are only ever added, never renamed or
# removed, so existing labels stay valid.
CATEGORIES: tuple[str, ...] = tuple(CATEGORY_KEYS.values())

CATEGORY_HINTS: dict[str, str] = {
    "native_compile": "kotlin/java/c++",
    "toolchain_version": "flutter/java/xcode version mismatch",
    "ios_build": "cocoapods/xcode",
    "infra_flaky": "network, runner, cache",
    "ci_config": "workflow, CI scripts, runner setup, secrets not passed",
}

CONFIDENCE_LEVELS = (1, 2, 3)

# Which evaluation set a label belongs to. Labels written before splits existed have no
# "split" field and are read as "dev": they were looked at while the prompt was tuned.
# New labels default to "test", held out to measure the tuned prompt.
DEV_SPLIT = "dev"
TEST_SPLIT = "test"
SPLITS = (DEV_SPLIT, TEST_SPLIT)


@dataclass(frozen=True)
class Label:
    job_id: int
    repo: str
    run_id: int
    category: str
    root_cause: str
    evidence_lines: list[int]
    confidence: int
    log_path: str
    labeled_at: str
    hint: str | None = None
    """Category the rule-based hint suggested, or None if there was no hint."""
    hint_accepted: bool | None = None
    """Whether `category` equals `hint` as first labeled; None when there was no hint.

    Not updated by later edits, so it keeps measuring the decision made with the hint on
    screen.
    """
    edited_at: str | None = None
    """When the label was last changed with `cidoc label edit`; None if never."""
    split: str = DEV_SPLIT
    """Evaluation set ("dev" or "test"); a missing field is read as "dev"."""


def label_from_record(record: dict[str, Any]) -> Label:
    """Build a Label, ignoring fields it doesn't know (e.g. added by a newer version).

    Raises TypeError if a required field is missing, AttributeError if not an object.
    """
    known = {f.name for f in fields(Label)}
    return Label(**{k: v for k, v in record.items() if k in known})


def load_labels(path: Path, on_bad_line: Callable[[int], None] | None = None) -> list[Label]:
    """Read every label. Unparseable lines (e.g. a write cut short by a crash) are skipped."""
    if not path.exists():
        return []
    labels: list[Label] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                labels.append(label_from_record(json.loads(line)))
            except (json.JSONDecodeError, TypeError, AttributeError):
                if on_bad_line:
                    on_bad_line(lineno)
    return labels


def append_label(path: Path, label: Label) -> None:
    """Append one label and fsync, so a crash right after never loses a finished label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# Fields `cidoc label edit` may change; everything else on the line is kept as is.
EDITABLE_FIELDS = ("category", "root_cause", "evidence_lines", "confidence")


class LabelNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class StoredLabel:
    line_index: int
    """0-based index of the line in labels.jsonl."""
    record: dict[str, Any]
    """The line's JSON object, with any fields this version doesn't know about."""

    @property
    def label(self) -> Label:
        return label_from_record(self.record)


def find_label(path: Path, job_id: int | None = None) -> StoredLabel:
    """The newest label for `job_id`, or the newest label overall when job_id is None.

    Raises LabelNotFoundError if there is none.
    """
    if not path.exists():
        raise LabelNotFoundError(f"no labels yet ({path} does not exist)")
    found: StoredLabel | None = None
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or "job_id" not in record:
                continue
            if job_id is None or record["job_id"] == job_id:
                found = StoredLabel(index, record)
    if found is None:
        what = "no labels" if job_id is None else f"no label for job {job_id}"
        raise LabelNotFoundError(f"{what} in {path}")
    return found


def replace_label_line(path: Path, line_index: int, record: dict[str, Any]) -> None:
    """Rewrite one line of labels.jsonl atomically; every other line is kept byte for byte.

    Writes a temp file in the same directory, fsyncs it, then renames it over the
    original, so a crash leaves either the old file or the new one, never a mix.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[line_index] = json.dumps(record, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(lines)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def parse_evidence(text: str, max_line: int) -> list[int]:
    """Parse '12, 40,41' into sorted unique line numbers. Raises ValueError if invalid."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("enter at least one line number")
    numbers: set[int] = set()
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"{part!r} is not a line number")
        number = int(part)
        if not 1 <= number <= max_line:
            raise ValueError(f"{number} is outside 1..{max_line}")
        numbers.add(number)
    return sorted(numbers)
