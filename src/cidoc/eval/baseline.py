"""The regex baseline: the labeling hint rules used as a model. No LLM, zero cost.

Category: the hint's category if a rule matches, else "other". Evidence: the line the
hint matched (the first match, as labelers saw it). It shares the LLM analyzer's result
type so the eval runner and metrics treat both the same way.
"""

from typing import Protocol

from cidoc.analyze.analyzer import AnalysisResult
from cidoc.analyze.inputs import AnalysisInput
from cidoc.analyze.schema import ROOT_CAUSE_MAX_CHARS, Diagnosis
from cidoc.labeling.hints import hint_for_log
from cidoc.logview import build_focus_view

REGEX_BASELINE = "regex-baseline"
FALLBACK_CATEGORY = "other"


class Predictor(Protocol):
    """Anything eval can score: one log in, one AnalysisResult out."""

    @property
    def name(self) -> str: ...

    def predict(self, source: AnalysisInput) -> AnalysisResult: ...


class RegexBaseline:
    @property
    def name(self) -> str:
        return REGEX_BASELINE

    def predict(self, source: AnalysisInput) -> AnalysisResult:
        hint = hint_for_log(source.text, build_focus_view(source.text, source.failed_steps))
        if hint is None:
            diagnosis = Diagnosis(
                category=FALLBACK_CATEGORY,
                root_cause="No hint rule matched.",
                evidence_lines=[],
                suggested_fix="(baseline: no fix suggested)",
                confidence=0.0,
            )
        else:
            diagnosis = Diagnosis(
                category=hint.category,
                root_cause=f"Hint rule matched: {hint.matched}"[:ROOT_CAUSE_MAX_CHARS],
                evidence_lines=[hint.line_no],
                suggested_fix="(baseline: no fix suggested)",
                confidence=1.0,
            )
        return AnalysisResult(model=REGEX_BASELINE, diagnosis=diagnosis, failure=None, calls=[])
