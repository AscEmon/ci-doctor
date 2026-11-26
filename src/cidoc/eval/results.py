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
from cidoc.labeling.labels import normalize_split

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
    split: str = "dev"
    """Labels scored: "dev", "test1", "test2" or "all". Runs saved before splits existed
    scored every label, and all of those are read as "dev"; a legacy "test" is "test1"."""
    agent: bool = False
    """A Phase 2 agent run (tools before answering); False for Phase 1 and older runs."""
    tools: list[str] | None = None
    """Agent runs: the tools offered (for ablations)."""
    max_steps: int | None = None
    """Agent runs: the tool-call limit per item."""


@dataclass(frozen=True)
class RunFile:
    info: RunInfo
    items: list[EvalItem]
    metrics: dict[str, Any]


def run_path(data_dir: Path, model: str, now: datetime, agent: bool = False) -> Path:
    """A new file name; never an existing run (two runs can finish in the same second,
    e.g. cached ablation runs back to back): "_2", "_3", ... are added as needed."""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{_UNSAFE.sub('_', model)}{'_agent' if agent else ''}"
    path = data_dir / RUNS_DIRNAME / f"{stem}.json"
    n = 2
    while path.exists():
        path = data_dir / RUNS_DIRNAME / f"{stem}_{n}.json"
        n += 1
    return path


def item_to_dict(item: EvalItem) -> dict[str, Any]:
    return {
        "job_id": item.job_id,
        "gold": item.gold,
        "gold_confidence": item.gold_confidence,
        "gold_evidence": item.gold_evidence,
        "gold_root_cause": item.gold_root_cause,
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
        gold_root_cause=data.get("gold_root_cause"),  # absent in runs saved before 3b
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
    info = replace(info, split=normalize_split(info.split))
    return RunFile(
        info=info,
        items=[item_from_dict(item) for item in data["items"]],
        metrics=data["metrics"],
    )
