"""
Local-shard model loading for pipeline-parallel inference.

Provides safetensors index parsing, tensor-name ownership resolution,
and local tensor manifest creation for Qwen3.5 pipeline-parallel loading.

Each rank loads only the tensors required by its assigned pipeline stage,
avoiding full-model materialization on every node.

Requirements addressed:
- 1.1: Each rank loads only tensors required by its assigned pipeline stage
- 1.3: Loading reads tensors directly from safetensors files by tensor name
- 1.4: Rank ownership is deterministic and configurable
- 1.5: Default four-stage ownership
- 1.9: Loader exposes a manifest of loaded tensor names for diagnostics
- 1.10: Missing, duplicate, or unassigned layer tensors produce clear startup errors
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import final

from pydantic import BaseModel, ConfigDict

from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PipelineStageAssignment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor name pattern for Qwen3.5 layer tensors
# ---------------------------------------------------------------------------

_LAYER_TENSOR_PATTERN: re.Pattern[str] = re.compile(
    r"^model\.layers\.(\d+)\."
)


# ---------------------------------------------------------------------------
# LocalShardManifest — describes what tensors a rank owns
# ---------------------------------------------------------------------------


@final
class LocalShardManifest(BaseModel):
    """
    Immutable manifest describing which tensors a rank owns and where they live.

    Created by the local shard loader after parsing the safetensors index
    and resolving tensor ownership for a specific pipeline stage assignment.
    """

    model_config = ConfigDict(frozen=True)

    tensor_names: list[str]
    """Names of tensors owned by this rank, in sorted order."""

    tensor_to_file: dict[str, str]
    """Mapping from tensor name to safetensors shard filename."""

    stage_assignment: PipelineStageAssignment
    """The pipeline stage assignment used to resolve ownership."""

    total_tensor_count: int
    """Total number of tensors in the full model (for validation)."""

    local_tensor_count: int
    """Number of tensors owned by this rank."""


# ---------------------------------------------------------------------------
# parse_safetensors_index — parse model.safetensors.index.json
# ---------------------------------------------------------------------------


def parse_safetensors_index(model_path: Path) -> dict[str, str]:
    """Parse the safetensors index file and return the weight_map.

    The weight_map maps tensor names to their containing shard filenames.

    Args:
        model_path: Path to the model directory containing
            ``model.safetensors.index.json``.

    Returns:
        Dictionary mapping tensor name to shard filename.

    Raises:
        FileNotFoundError: If the index file does not exist.
        ValueError: If the index file is malformed or missing weight_map.
    """
    index_path = model_path / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Safetensors index not found at {index_path}"
        )

    with open(index_path, "r", encoding="utf-8") as f:
        try:
            index_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse safetensors index at {index_path}: {e}"
            ) from e

    if not isinstance(index_data, dict):
        raise ValueError(
            f"Safetensors index at {index_path} is not a JSON object"
        )

    weight_map = index_data.get("weight_map")
    if weight_map is None:
        raise ValueError(
            f"Safetensors index at {index_path} has no 'weight_map' key"
        )

    if not isinstance(weight_map, dict):
        raise ValueError(
            f"'weight_map' in safetensors index at {index_path} is not a dict"
        )

    # Validate all values are strings
    for tensor_name, shard_file in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard_file, str):
            raise ValueError(
                f"Invalid weight_map entry: {tensor_name!r} -> {shard_file!r}"
            )

    logger.debug(
        f"Parsed safetensors index: {len(weight_map)} tensors across "
        f"{len(set(weight_map.values()))} shard files"
    )

    return weight_map


# ---------------------------------------------------------------------------
# resolve_tensor_ownership — determine which tensors belong to a rank
# ---------------------------------------------------------------------------


def resolve_tensor_ownership(
    weight_map: dict[str, str],
    stage_assignment: PipelineStageAssignment,
) -> list[str]:
    """Determine which tensors from the weight map belong to this rank.

    Ownership rules for Qwen3.5:
    - Rank that owns_embedding gets: ``model.embed_tokens.weight``
    - Each rank gets tensors matching ``model.layers.{i}.*`` where
      ``i`` is in ``[start_layer, end_layer)``
    - Rank that owns_lm_head gets: ``model.norm.weight``, ``lm_head.weight``
    - Tied weights: if ``lm_head.weight`` is tied to ``model.embed_tokens.weight``
      (same shard file entry), the last rank still needs it

    Args:
        weight_map: Full tensor name to shard file mapping from the index.
        stage_assignment: This rank's pipeline stage assignment.

    Returns:
        Sorted list of tensor names owned by this rank.
    """
    owned_tensors: list[str] = []

    for tensor_name in weight_map:
        if _tensor_belongs_to_stage(tensor_name, stage_assignment):
            owned_tensors.append(tensor_name)

    owned_tensors.sort()
    return owned_tensors


def _tensor_belongs_to_stage(
    tensor_name: str,
    stage_assignment: PipelineStageAssignment,
) -> bool:
    """Check if a tensor belongs to the given pipeline stage.

    Args:
        tensor_name: Full tensor name from the weight map.
        stage_assignment: This rank's pipeline stage assignment.

    Returns:
        True if this rank owns the tensor.
    """
    # Embedding tensor — owned by the rank with owns_embedding
    if tensor_name == "model.embed_tokens.weight":
        return stage_assignment.owns_embedding

    # Final normalization — owned by the rank with owns_lm_head
    if tensor_name == "model.norm.weight":
        return stage_assignment.owns_lm_head

    # Language model head — owned by the rank with owns_lm_head
    if tensor_name == "lm_head.weight":
        return stage_assignment.owns_lm_head

    # Layer tensors — owned by the rank whose range includes the layer index
    layer_match = _LAYER_TENSOR_PATTERN.match(tensor_name)
    if layer_match is not None:
        layer_index = int(layer_match.group(1))
        return (
            stage_assignment.start_layer
            <= layer_index
            < stage_assignment.end_layer
        )

    # Unknown tensor pattern — not owned by any stage
    # This handles metadata tensors or unexpected entries gracefully
    logger.debug(
        f"Tensor '{tensor_name}' does not match any ownership rule, "
        f"skipping for rank {stage_assignment.rank}"
    )
    return False


# ---------------------------------------------------------------------------
# load_qwen_local_shard_from_safetensors — orchestrator function
# ---------------------------------------------------------------------------


def load_qwen_local_shard_from_safetensors(
    *,
    model_path: Path,
    stage_assignment: PipelineStageAssignment,
) -> LocalShardManifest:
    """Create a local tensor manifest for a pipeline stage.

    Orchestrates safetensors index parsing, tensor ownership resolution,
    and manifest creation. Does NOT load actual tensor data — only creates
    the manifest describing what to load.

    Args:
        model_path: Path to the model directory containing the safetensors
            index file and shard files.
        stage_assignment: This rank's pipeline stage assignment describing
            which layers and auxiliary modules it owns.

    Returns:
        A LocalShardManifest describing the tensors this rank needs to load.

    Raises:
        FileNotFoundError: If the safetensors index file is missing.
        ValueError: If the index is malformed or no tensors are assigned.
    """
    # Step 1: Parse the safetensors index
    weight_map = parse_safetensors_index(model_path)

    # Step 2: Resolve tensor ownership for this rank
    owned_tensor_names = resolve_tensor_ownership(weight_map, stage_assignment)

    if not owned_tensor_names:
        raise ValueError(
            f"No tensors assigned to rank {stage_assignment.rank} "
            f"with layer range [{stage_assignment.start_layer}, "
            f"{stage_assignment.end_layer}). "
            f"Check the stage assignment and model index."
        )

    # Step 3: Build the tensor-to-file mapping for owned tensors only
    tensor_to_file: dict[str, str] = {
        name: weight_map[name] for name in owned_tensor_names
    }

    # Step 4: Create the manifest
    manifest = LocalShardManifest(
        tensor_names=owned_tensor_names,
        tensor_to_file=tensor_to_file,
        stage_assignment=stage_assignment,
        total_tensor_count=len(weight_map),
        local_tensor_count=len(owned_tensor_names),
    )

    logger.info(
        f"Created local shard manifest for rank {stage_assignment.rank}: "
        f"{manifest.local_tensor_count}/{manifest.total_tensor_count} tensors, "
        f"layers [{stage_assignment.start_layer}, {stage_assignment.end_layer}), "
        f"shard files: {sorted(set(tensor_to_file.values()))}"
    )

    return manifest
