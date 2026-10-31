"""Failure taxonomy and the append-only labels.jsonl store."""

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

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
    "o": "other",
}

CATEGORY_HINTS: dict[str, str] = {
    "native_compile": "kotlin/java/c++",
    "toolchain_version": "flutter/java/xcode version mismatch",
    "ios_build": "cocoapods/xcode",
    "infra_flaky": "network, runner, cache",
}

CONFIDENCE_LEVELS = (1, 2, 3)


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
                labels.append(Label(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
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
