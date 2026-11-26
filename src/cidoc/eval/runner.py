"""Run a model over the labeled jobs.

Resumable: the LLM predictor goes through the response cache, so items scored by an
earlier (possibly interrupted) run cost nothing and make no API call.

One failing item never stops the run: an error that survives the provider's backoff is
recorded on that item (not cached, so the next run retries it) and the run moves on.
Only the budget (--max-cost) and Ctrl-C end a run early; the caller saves what was
scored.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cidoc.agent.budget import BudgetedToolProvider, estimate_item_calls
from cidoc.agent.loop import agent_cache_key, run_agent_cached
from cidoc.agent.runtime import AgentSetup, job_context, make_toolbox
from cidoc.agent.tools import GitHubReader
from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.cache import ResponseCache, analyze_cached, cache_key, log_hash
from cidoc.analyze.costguard import BudgetExceededError, CallEstimate, estimate_call
from cidoc.analyze.inputs import AnalysisInput, InputError, from_job_id
from cidoc.analyze.preprocess import PreprocessedLog, preprocess
from cidoc.analyze.prompt import Prompt
from cidoc.eval.baseline import Predictor
from cidoc.eval.metrics import EvalItem
from cidoc.labeling.labels import DEV_SPLIT, LABELS_FILENAME, Label, load_labels, newest_labels
from cidoc.llm.base import LLMProvider, ToolCallingProvider
from cidoc.llm.pricing import Price

ALL_SPLITS = "all"
"""`cidoc eval --split all`: every label, whatever its split."""


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
    data_dir: Path,
    open_db: Callable[[], sqlite3.Connection],
    limit: int | None = None,
    split: str = DEV_SPLIT,
) -> Cases:
    """Every non-skipped label (newest per job) in `split`, in file order, with its stored log.

    `split` is "dev", "test1", "test2" or "all"; the newest label decides a job's split. Skips are
    never written to labels.jsonl; a record whose category is not in the taxonomy is
    treated as skipped too.
    """
    newest = newest_labels(load_labels(data_dir / LABELS_FILENAME))
    labels = [lbl for lbl in newest.values() if split in (ALL_SPLITS, lbl.split)]
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


@dataclass(frozen=True)
class RunPlan:
    calls: list[CallEstimate]
    """Estimated model calls of the uncached cases, in run order (one per case for
    Phase 1; up to max_steps + 1 per case for the agent)."""
    cached: int


def plan_run(
    cases: list[EvalCase],
    prompt: Prompt,
    cache: ResponseCache,
    model: str,
    token_budget: int,
    refresh: bool = False,
) -> RunPlan:
    """What an LLM run would send, without a provider or any API call.

    Cached cases are free; every other case is estimated as one call.
    """
    calls: list[CallEstimate] = []
    cached = 0
    for case in cases:
        log = preprocess(case.source.text, case.source.failed_steps, token_budget)
        key = cache_key(prompt.version, log_hash(log), model)
        if not refresh and cache.get(key) is not None:
            cached += 1
        else:
            calls.append(estimate_call(prompt.system, prompt.user(log)))
    return RunPlan(calls, cached)


class AgentPredictor:
    """The Phase 2 agent as an eval model. Same excerpt as Phase 1, plus tools bound to
    each job's run; results go through the response cache (resumable) and tool results
    through the tool cache (re-runs make no GitHub calls for them)."""

    def __init__(
        self,
        provider: ToolCallingProvider,
        setup: AgentSetup,
        cache: ResponseCache,
        token_budget: int,
        data_dir: Path,
        open_db: Callable[[], sqlite3.Connection],
        github: GitHubReader | None,
        price: Price | None = None,
        refresh: bool = False,
    ) -> None:
        self.provider = provider
        self.setup = setup
        self.cache = cache
        self.token_budget = token_budget
        self.data_dir = data_dir
        self.open_db = open_db
        self.github = github
        self.price = price
        self.refresh = refresh

    @property
    def name(self) -> str:
        return self.provider.model

    def predict(self, source: AnalysisInput) -> AnalysisResult:
        if source.job_id is None:
            raise InputError("the agent needs a stored job id")
        log = preprocess(source.text, source.failed_steps, self.token_budget)
        ctx = job_context(source.job_id, source.text, self.open_db)
        if isinstance(self.provider, BudgetedToolProvider):
            key = agent_cache_key(self.setup.prompt, log, ctx, self.provider.model)
            if self.refresh or self.cache.get(key) is None:
                # Do not start an item whose worst case would pass --max-cost.
                self.provider.check_item(
                    estimate_item_calls(self.setup.prompt, log, self.setup.limits)
                )
        tools = make_toolbox(ctx, self.github, self.setup, self.data_dir)
        return run_agent_cached(
            self.provider,
            log,
            self.setup.prompt,
            tools,
            self.cache,
            self.setup.limits,
            price=self.price,
            refresh=self.refresh,
        )


def plan_agent_run(
    cases: list[EvalCase],
    setup: AgentSetup,
    cache: ResponseCache,
    model: str,
    token_budget: int,
    open_db: Callable[[], sqlite3.Connection],
    refresh: bool = False,
) -> RunPlan:
    """What an agent run would send, without a provider or any API call.

    Cached cases are free. Every other case is estimated at its worst case: max_steps
    tool rounds and the final submit, each call resending the conversation so far.
    """
    calls: list[CallEstimate] = []
    cached = 0
    for case in cases:
        log = preprocess(case.source.text, case.source.failed_steps, token_budget)
        job_id = case.source.job_id if case.source.job_id is not None else case.label.job_id
        ctx = job_context(job_id, case.source.text, open_db)
        key = agent_cache_key(setup.prompt, log, ctx, model)
        if not refresh and cache.get(key) is not None:
            cached += 1
        else:
            calls.extend(estimate_item_calls(setup.prompt, log, setup.limits))
    return RunPlan(calls, cached)


ERROR_MAX_CHARS = 300


def to_item(case: EvalCase, result: AnalysisResult, error: str | None = None) -> EvalItem:
    return EvalItem(
        job_id=case.label.job_id,
        gold=case.label.category,
        gold_confidence=case.label.confidence,
        gold_evidence=list(case.label.evidence_lines),
        result=result,
        error=error,
        gold_root_cause=case.label.root_cause,
    )


def errored_item(case: EvalCase, model: str, exc: Exception) -> EvalItem:
    error = f"{type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
    return to_item(case, AnalysisResult(model, None, None, calls=[]), error)


@dataclass(frozen=True)
class RunOutcome:
    items: list[EvalItem]
    stopped: str | None
    """Why the run ended before the last case; None when it finished."""
    interrupted: bool = False
    """Stopped by Ctrl-C."""

    @property
    def errored(self) -> int:
        return sum(item.error is not None for item in self.items)


def run_eval(
    cases: list[EvalCase],
    predictor: Predictor,
    on_item: Callable[[int, EvalItem], None] = lambda n, item: None,
    stop_on: tuple[type[Exception], ...] = (BudgetExceededError,),
) -> RunOutcome:
    """Score each case in order.

    A `stop_on` error or Ctrl-C ends the run with what was scored. Any other error is
    recorded on its item, which is not scored, and the run continues.
    """
    items: list[EvalItem] = []
    for n, case in enumerate(cases, start=1):
        try:
            item = to_item(case, predictor.predict(case.source))
        except stop_on as exc:
            return RunOutcome(items, f"job {case.label.job_id}: {exc}")
        except KeyboardInterrupt:
            return RunOutcome(items, "interrupted (Ctrl-C)", interrupted=True)
        except Exception as exc:  # one bad item must not stop the run
            item = errored_item(case, predictor.name, exc)
        items.append(item)
        on_item(n, item)
    return RunOutcome(items, None)
