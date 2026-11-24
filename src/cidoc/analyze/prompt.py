"""Prompt text for the analyzer.

Minimal for now; the injection-safe system instruction with category definitions
replaces it.
"""

from cidoc.analyze.preprocess import PreprocessedLog

SYSTEM_INSTRUCTION = (
    "You diagnose failed GitHub Actions jobs of Flutter projects. Read the log excerpt, "
    "find the root cause, and answer with a JSON object matching the response schema. "
    "Lines are prefixed 'L<number>| ' with their number in the original log; cite those "
    "numbers as evidence_lines."
)


def user_prompt(log: PreprocessedLog) -> str:
    return f"Failed steps: {', '.join(log.failed_titles) or 'unknown'}\n\n{log.text}"
