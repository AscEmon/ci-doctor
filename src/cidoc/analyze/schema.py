"""What the analyzer must return: a JSON schema for the model and a pydantic model to check it.

The category enum comes from the labeling taxonomy (labels.CATEGORIES), the single source.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from cidoc.labeling.labels import CATEGORIES

ROOT_CAUSE_MAX_CHARS = 200

# Sent to the model as response_json_schema. Kept to widely supported keywords (type,
# properties, required, enum, items, minimum, maximum, description); Diagnosis enforces
# the rest.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": list(CATEGORIES),
            "description": "The single taxonomy category of the root cause.",
        },
        "root_cause": {
            "type": "string",
            "description": f"One sentence, at most {ROOT_CAUSE_MAX_CHARS} characters.",
        },
        "evidence_lines": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Original log line numbers (the N in 'LN|') that show the cause.",
        },
        "suggested_fix": {"type": "string", "description": "A concrete next step."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["category", "root_cause", "evidence_lines", "suggested_fix", "confidence"],
}


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    root_cause: str = Field(min_length=1, max_length=ROOT_CAUSE_MAX_CHARS)
    evidence_lines: list[PositiveInt]
    suggested_fix: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"must be one of: {', '.join(CATEGORIES)}")
        return value

    @field_validator("root_cause", "suggested_fix")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()
