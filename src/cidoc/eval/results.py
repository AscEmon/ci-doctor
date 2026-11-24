"""Eval run files: data/eval_runs/{timestamp}_{model}.json.

Each file records what produced it (model, prompt version, guideline hash, token budget),
every item's prediction next to its label, and the metrics. Items hold enough to
recompute the metrics, so runs can be compared later.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.cache import result_from_dict, result_to_dict
from cidoc.eval.metrics import EvalItem, Metrics

RUNS_DIRNAME = "eval_runs"
FORMAT_VERSION = 1
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RunInfo:
    model: str
    created_at: str
    prompt_version: str | None
    """None for models without a prompt (the regex baseline)."""
    guideline_hash: str
    token_budget: int | None
    labels_total: int
    complete: bool
    stopped: str | None = None
    """Why the run ended early; None when every item was scored."""
    missing: list[int] = field(default_factory=list)
    """Labeled jobs whose log is gone; not scored."""


@dataclass(frozen=True)
class RunFile:
    info: RunInfo
    items: list[EvalItem]
    metrics: dict[str, Any]


def run_path(data_dir: Path, model: str, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return data_dir / RUNS_DIRNAME / f"{stamp}_{_UNSAFE.sub('_', model)}.json"


def item_to_dict(item: EvalItem) -> dict[str, Any]:
    return {
        "job_id": item.job_id,
        "gold": item.gold,
        "gold_confidence": item.gold_confidence,
        "gold_evidence": item.gold_evidence,
        "predicted": item.predicted,
        "correct": item.correct,
        "cached": item.result.cached,
        "error": item.error,
        "result": result_to_dict(item.result),
    }


def item_from_dict(data: dict[str, Any]) -> EvalItem:
    result: AnalysisResult = replace(result_from_dict(data["result"]), cached=data["cached"])
    return EvalItem(
        job_id=data["job_id"],
        gold=data["gold"],
        gold_confidence=data["gold_confidence"],
        gold_evidence=list(data["gold_evidence"]),
        result=result,
        error=data.get("error"),  # absent in runs saved before errored items existed
    )


def write_run(path: Path, info: RunInfo, items: list[EvalItem], metrics: Metrics) -> None:
    """Atomic: a run file is either complete JSON or absent."""
    payload = {
        "format": FORMAT_VERSION,
        **asdict(info),
        "metrics": asdict(metrics),
        "items": [item_to_dict(item) for item in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_run(path: Path) -> RunFile:
    data = json.loads(path.read_text(encoding="utf-8"))
    info_fields = set(RunInfo.__dataclass_fields__)
    info = RunInfo(**{k: v for k, v in data.items() if k in info_fields})
    return RunFile(
        info=info,
        items=[item_from_dict(item) for item in data["items"]],
        metrics=data["metrics"],
    )
