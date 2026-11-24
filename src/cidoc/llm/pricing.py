"""Token prices. Values come from configuration (CIDOC_PRICES), never from code."""

from pydantic import BaseModel, ConfigDict, NonNegativeFloat

PER_MILLION = 1_000_000


class Price(BaseModel):
    """USD per million tokens. Thinking tokens are billed as output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: NonNegativeFloat
    output: NonNegativeFloat

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / PER_MILLION
