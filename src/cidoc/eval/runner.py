"""Run a model over the labeled jobs.

Resumable: the LLM predictor goes through the response cache, so items scored by an
earlier (possibly interrupted) run cost nothing and make no API call. An API error that
survives the provider's backoff stops the run cleanly; the items scored so far are kept
and the caller saves them.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.cache import ResponseCache, analyze_cached
from cidoc.analyze.inputs import AnalysisInput, InputError, from_job_id
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import Prompt
from cidoc.eval.baseline import Predictor
from cidoc.eval.metrics import EvalItem
from cidoc.labeling.labels import CATEGORIES, LABELS_FILENAME, Label, load_labels
from cidoc.llm.base import LLMError, LLMProvider


@dataclass(frozen=True)
class EvalCase:
    label: Label
    source: AnalysisInput


@dataclass(frozen=True)
class Cases:
    cases: list[EvalCase]
    labels_total: int
    missing: list[int] = field(default_factory=list)
    """Labeled jobs whose log could not be read."""


def load_cases(
    data_dir: Path, open_db: Callable[[], sqlite3.Connection], limit: int | None = None
) -> Cases:
    """Every non-skipped label (newest per job), in file order, with its stored log.

    Skips are never written to labels.jsonl; a record whose category is not in the
    taxonomy is treated as skipped too.
    """
    newest: dict[int, Label] = {}
    for label in load_labels(data_dir / LABELS_FILENAME):
        if label.category in CATEGORIES:
            newest.pop(label.job_id, None)  # keep file order of the newest
            newest[label.job_id] = label
    labels = list(newest.values())
    if limit is not None:
        labels = labels[:limit]
    cases: list[EvalCase] = []
    missing: list[int] = []
    for label in labels:
        try:
            cases.append(EvalCase(label, from_job_id(label.job_id, data_dir, open_db)))
        except InputError:
            missing.append(label.job_id)
    return Cases(cases, labels_total=len(labels), missing=missing)


class LLMPredictor:
    def __init__(
        self,
        provider: LLMProvider,
        prompt: Prompt,
        cache: ResponseCache,
        token_budget: int,
        refresh: bool = False,
    ) -> None:
        self.provider = provider
        self.prompt = prompt
        self.cache = cache
        self.token_budget = token_budget
        self.refresh = refresh

    @property
    def name(self) -> str:
        return self.provider.model

    def preprocess(self, source: AnalysisInput) -> PreprocessedLog:
        return preprocess(source.text, source.failed_steps, self.token_budget)

    def predict(self, source: AnalysisInput) -> AnalysisResult:
        log = self.preprocess(source)
        return analyze_cached(self.provider, log, self.prompt, self.cache, refresh=self.refresh)


def to_item(case: EvalCase, result: AnalysisResult) -> EvalItem:
    return EvalItem(
        job_id=case.label.job_id,
        gold=case.label.category,
        gold_confidence=case.label.confidence,
        gold_evidence=list(case.label.evidence_lines),
        result=result,
    )


@dataclass(frozen=True)
class RunOutcome:
    items: list[EvalItem]
    stopped: str | None
    """Why the run ended before the last case; None when it finished."""


def run_eval(
    cases: list[EvalCase],
    predictor: Predictor,
    on_item: Callable[[int, EvalItem], None] = lambda n, item: None,
    stop_on: tuple[type[Exception], ...] = (LLMError,),
) -> RunOutcome:
    """Score each case in order; `stop_on` errors end the run with what was scored."""
    items: list[EvalItem] = []
    for n, case in enumerate(cases, start=1):
        try:
            result = predictor.predict(case.source)
        except stop_on as exc:
            return RunOutcome(items, f"job {case.label.job_id}: {exc}")
        item = to_item(case, result)
        items.append(item)
        on_item(n, item)
    return RunOutcome(items, None)
