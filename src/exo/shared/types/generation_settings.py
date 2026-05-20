from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel


class GenerationSettings(FrozenModel):
    """Cluster-wide inference parameter defaults."""

    thinking_mode: bool = False
    thinking_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
    output_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
