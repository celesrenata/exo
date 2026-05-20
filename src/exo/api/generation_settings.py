"""Generation settings API routes.

Provides GET and PATCH endpoints for managing generation settings.
- GET /api/generation/settings: Returns current generation settings.
- PATCH /api/generation/settings: Updates generation settings with partial updates.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from exo.shared.types.commands import UpdateGenerationSettings
from exo.shared.types.generation_settings import GenerationSettings

if TYPE_CHECKING:
    from exo.api.main import API


class GenerationSettingsPatch(BaseModel):
    """Partial update model for generation settings.

    All fields are optional. Only fields present in the request body are applied.
    Uses exclude_unset=True to distinguish absent fields from explicit nulls.
    """

    thinking_mode: bool | None = None
    thinking_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)
    output_token_budget: int | None = Field(default=None, ge=1, le=1_048_576)


def _merge_settings(
    current: GenerationSettings, patch: GenerationSettingsPatch
) -> GenerationSettings:
    """Merge a partial patch into the current settings, returning a new instance."""
    current_dict = current.model_dump()
    patch_dict = patch.model_dump(exclude_unset=True)
    merged_dict = {**current_dict, **patch_dict}
    return GenerationSettings.model_validate(merged_dict)


def register_generation_settings_routes(api: "API") -> None:
    """Register GET and PATCH /api/generation/settings on the API app."""

    @api.app.get("/api/generation/settings")
    async def get_generation_settings() -> GenerationSettings:  # pyright: ignore[reportUnusedFunction]
        return api.state.generation_settings

    @api.app.patch("/api/generation/settings")
    async def patch_generation_settings(  # pyright: ignore[reportUnusedFunction]
        patch: GenerationSettingsPatch,
    ) -> GenerationSettings:
        new_settings = _merge_settings(api.state.generation_settings, patch)
        await api._send(UpdateGenerationSettings(generation_settings=new_settings))  # pyright: ignore[reportPrivateUsage]
        return new_settings
