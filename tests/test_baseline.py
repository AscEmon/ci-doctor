from cidoc.analyze.inputs import AnalysisInput
from cidoc.analyze.usage import summarize
from cidoc.eval.baseline import REGEX_BASELINE, RegexBaseline
from cidoc.labeling.hints import hint_for_log
from cidoc.llm.pricing import Price
from cidoc.logview import build_focus_view

TS = "2026-10-01T12:34:56.1234567Z "


def source(*lines: str, failed_steps: list[str] | None = None) -> AnalysisInput:
    text = "".join(f"{TS}{line}\n" for line in lines)
    return AnalysisInput(text=text, source="test", failed_steps=failed_steps or [])


PUB_FAILURE = source(
    "##[group]Run flutter pub get",  # 1
    "##[endgroup]",  # 2
    "Resolving dependencies...",  # 3
    "Because app depends on intl ^9.0.0 which doesn't match any versions,"
    " version solving failed.",  # 4
    "##[error]Process completed with exit code 1.",  # 5
)


def test_matching_hint_gives_its_category_and_line() -> None:
    result = RegexBaseline().predict(PUB_FAILURE)

    assert result.ok
    assert result.model == REGEX_BASELINE
    assert result.diagnosis is not None
    assert result.diagnosis.category == "dependency_resolution"
    assert result.diagnosis.evidence_lines == [4]
    assert "version solving failed" in result.diagnosis.root_cause
    assert result.diagnosis.confidence == 1.0


def test_no_hint_means_other_without_evidence() -> None:
    result = RegexBaseline().predict(
        source(
            "##[group]Run ./scripts/release.sh",
            "##[endgroup]",
            "something unusual happened",
            "##[error]Process completed with exit code 3.",
        )
    )

    assert result.diagnosis is not None
    assert result.diagnosis.category == "other"
    assert result.diagnosis.evidence_lines == []
    assert result.diagnosis.confidence == 0.0


def test_agrees_with_the_labeling_hint() -> None:
    hint = hint_for_log(PUB_FAILURE.text, build_focus_view(PUB_FAILURE.text))
    result = RegexBaseline().predict(PUB_FAILURE)

    assert hint is not None
    assert result.diagnosis is not None
    assert (result.diagnosis.category, result.diagnosis.evidence_lines) == (
        hint.category,
        [hint.line_no],
    )


def test_only_the_failed_step_is_searched() -> None:
    # A keystore mention in a successful setup step is not why the job failed.
    result = RegexBaseline().predict(
        source(
            "##[group]Run echo setup",  # 1
            "##[endgroup]",  # 2
            "Keystore file 'upload.jks' not found for signing config 'release'.",  # 3
            "##[group]Run flutter pub get",  # 4
            "##[endgroup]",  # 5
            "Because app depends on intl ^9.0.0 which doesn't match any versions,"
            " version solving failed.",  # 6
            "##[error]Process completed with exit code 1.",  # 7
        )
    )

    assert result.diagnosis is not None
    assert result.diagnosis.category == "dependency_resolution"
    assert result.diagnosis.evidence_lines == [6]


def test_costs_nothing() -> None:
    result = RegexBaseline().predict(PUB_FAILURE)

    assert result.calls == []
    assert summarize(result.calls, Price(input=1, output=1)).cost == 0.0
    assert not result.cached


def test_long_matches_fit_the_root_cause_limit() -> None:
    long_line = "version solving failed. " + "x" * 400
    result = RegexBaseline().predict(
        source("##[group]Run flutter pub get", "##[endgroup]", long_line, "##[error]exit 1")
    )

    assert result.diagnosis is not None
    assert len(result.diagnosis.root_cause) <= 200
