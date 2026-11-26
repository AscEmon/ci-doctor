"""Response cache: re-analyzing the same log with the same prompt and model is free.

Keyed by (prompt version, log hash, model). The log hash covers exactly the data sent
(failed step names and preprocessed text), so a change to preprocessing or the token
budget is a new key. Results without a diagnosis are cached too: a re-run must never pay
again for an answer that already failed validation twice. API errors are not cached.

One JSON file per key under <data_dir>/cache/analyze/ (data/ is never committed).
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cidoc.agent.trajectory import trajectory_from_dict, trajectory_to_dict
from cidoc.analyze.analyzer import AnalysisResult, analyze_log
from cidoc.analyze.preprocess import PreprocessedLog
from cidoc.analyze.prompt import Prompt
from cidoc.analyze.schema import AgentDiagnosis, Diagnosis
from cidoc.llm.base import LLMProvider, Usage

CACHE_DIRNAME = "cache/analyze"
FORMAT_VERSION = 1


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def log_hash(log: PreprocessedLog) -> str:
    return _sha256(json.dumps([log.failed_titles, log.text]))


def cache_key(prompt_version: str, log_digest: str, model: str) -> str:
    return _sha256(json.dumps([FORMAT_VERSION, prompt_version, log_digest, model]))


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": result.model,
        "diagnosis": result.diagnosis.model_dump() if result.diagnosis else None,
        "failure": result.failure,
        "calls": [asdict(call) for call in result.calls],
        "outputs": result.outputs,
    }
    if result.trajectory is not None:  # agent runs only; Phase 1 entries are unchanged
        data["trajectory"] = trajectory_to_dict(result.trajectory)
    return data


def _diagnosis_from_dict(data: dict[str, Any]) -> Diagnosis:
    if "evidence_files" in data:
        return AgentDiagnosis.model_validate(data)
    return Diagnosis.model_validate(data)


def result_from_dict(data: dict[str, Any]) -> AnalysisResult:
    diagnosis = data["diagnosis"]
    trajectory = data.get("trajectory")
    return AnalysisResult(
        model=data["model"],
        diagnosis=_diagnosis_from_dict(diagnosis) if diagnosis is not None else None,
        failure=data["failure"],
        calls=[Usage(**call) for call in data["calls"]],
        outputs=list(data.get("outputs", [])),  # absent in entries cached before they were kept
        trajectory=trajectory_from_dict(trajectory) if trajectory is not None else None,
    )


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def in_data_dir(cls, data_dir: Path) -> "ResponseCache":
        return cls(data_dir / CACHE_DIRNAME)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> AnalysisResult | None:
        """The cached result, or None (also for an unreadable or outdated entry)."""
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            return replace(result_from_dict(entry["result"]), cached=True)
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def put(self, key: str, result: AnalysisResult, meta: dict[str, str]) -> None:
        """Write atomically, so an interrupted run never leaves a half-written entry."""
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, **meta, "result": result_to_dict(result)}
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=1)
            os.replace(tmp, self._path(key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def analyze_cached(
    provider: LLMProvider,
    log: PreprocessedLog,
    prompt: Prompt,
    cache: ResponseCache,
    refresh: bool = False,
) -> AnalysisResult:
    """Cached result if there is one (unless `refresh`), else analyze and store it."""
    digest = log_hash(log)
    key = cache_key(prompt.version, digest, provider.model)
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit
    result = analyze_log(provider, log, prompt)
    meta = {
        "prompt_version": prompt.version,
        "guideline_hash": prompt.guideline_hash,
        "log_hash": digest,
        "model": provider.model,
    }
    cache.put(key, result, meta)
    return result
