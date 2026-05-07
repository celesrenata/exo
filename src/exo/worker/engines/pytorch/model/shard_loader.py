"""Pipeline-stage-aware partial model loading.

Filters a full model state_dict to include only the tensors assigned to
the local pipeline stage, then moves them to the target GPU device.

Transformer layer naming convention:
  - Layers: ``model.layers.{N}.`` where N is the layer index
  - Embedding: ``model.embed_tokens.`` prefix (rank 0 only)
  - LM head: ``lm_head.`` prefix (last rank only)
  - Layer norm: ``model.norm.`` prefix (last rank, alongside lm_head)

Requirements: 10.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from exo.worker.engines.pytorch.pipeline.stage import StageAssignment

logger = logging.getLogger(__name__)

# Naming convention prefixes
_LAYER_PREFIX = "model.layers."
_EMBEDDING_PREFIX = "model.embed_tokens."
_LM_HEAD_PREFIX = "lm_head."
_LAYER_NORM_PREFIX = "model.norm."


@dataclass(frozen=True)
class ShardLoadResult:
    """Result of a pipeline-stage-aware partial model load.

    Attributes:
        tensor_count: Total number of tensors loaded for this stage.
        layer_count: Number of transformer layers loaded.
        has_embedding: Whether embedding tensors were loaded.
        has_lm_head: Whether LM head tensors were loaded.
        has_layer_norm: Whether final layer norm tensors were loaded.
        device: Target device string (e.g. "xpu:0", "cuda:0").
    """

    tensor_count: int
    layer_count: int
    has_embedding: bool
    has_lm_head: bool
    has_layer_norm: bool
    device: str


def _extract_layer_index(key: str) -> int | None:
    """Extract the layer index from a state_dict key.

    Returns None if the key does not match the transformer layer pattern.
    """
    if not key.startswith(_LAYER_PREFIX):
        return None
    # key looks like "model.layers.7.self_attn.q_proj.weight"
    rest = key[len(_LAYER_PREFIX):]
    dot_pos = rest.find(".")
    if dot_pos == -1:
        return None
    try:
        return int(rest[:dot_pos])
    except ValueError:
        return None


def filter_state_dict_for_stage(
    state_dict: dict[str, "torch.Tensor"],
    stage_assignment: StageAssignment,
) -> dict[str, "torch.Tensor"]:
    """Filter a full state_dict to only include tensors for the assigned stage.

    Includes:
      - Transformer layers in [start_layer, end_layer)
      - Embedding tensors if stage has_embedding (rank 0)
      - LM head and final layer norm tensors if stage has_lm_head (last rank)

    Args:
        state_dict: Complete model state_dict with all layers.
        stage_assignment: The pipeline stage assignment for this rank.

    Returns:
        Filtered state_dict containing only tensors for this stage.
    """
    filtered: dict[str, "torch.Tensor"] = {}

    for key, tensor in state_dict.items():
        # Check if it's a transformer layer
        layer_idx = _extract_layer_index(key)
        if layer_idx is not None:
            if stage_assignment.start_layer <= layer_idx < stage_assignment.end_layer:
                filtered[key] = tensor
            continue

        # Check embedding
        if key.startswith(_EMBEDDING_PREFIX):
            if stage_assignment.has_embedding:
                filtered[key] = tensor
            continue

        # Check LM head
        if key.startswith(_LM_HEAD_PREFIX):
            if stage_assignment.has_lm_head:
                filtered[key] = tensor
            continue

        # Check final layer norm (loaded with lm_head on last rank)
        if key.startswith(_LAYER_NORM_PREFIX):
            if stage_assignment.has_lm_head:
                filtered[key] = tensor
            continue

        # Unknown prefix — include on rank 0 as a fallback for misc model tensors
        if stage_assignment.has_embedding:
            filtered[key] = tensor

    logger.info(
        "Filtered state_dict for rank %d: %d/%d tensors (layers %d-%d)",
        stage_assignment.rank,
        len(filtered),
        len(state_dict),
        stage_assignment.start_layer,
        stage_assignment.end_layer,
    )

    return filtered


def move_state_dict_to_device(
    state_dict: dict[str, "torch.Tensor"],
    device: str,
) -> dict[str, "torch.Tensor"]:
    """Move all tensors in a state_dict to the target device.

    Args:
        state_dict: Dictionary of tensor name → tensor.
        device: Target device string (e.g. "xpu:0", "cuda:0").

    Returns:
        New state_dict with all tensors on the target device.
    """
    moved: dict[str, "torch.Tensor"] = {}
    for key, tensor in state_dict.items():
        moved[key] = tensor.to(device)
    logger.debug("Moved %d tensors to %s", len(moved), device)
    return moved


def load_shard_for_stage(
    state_dict: dict[str, "torch.Tensor"],
    stage_assignment: StageAssignment,
    device_type: Literal["cuda", "xpu"],
    device_index: int = 0,
) -> tuple[dict[str, "torch.Tensor"], ShardLoadResult]:
    """Filter and move a state_dict for a pipeline stage.

    This is the primary entry point for pipeline-stage-aware loading.
    It filters the full state_dict to only include tensors for the
    assigned stage, moves them to the target GPU, and validates the result.

    Args:
        state_dict: Complete model state_dict.
        stage_assignment: Pipeline stage assignment for this rank.
        device_type: Target device type ("cuda" or "xpu").
        device_index: Target device index (default 0).

    Returns:
        Tuple of (filtered_state_dict_on_device, ShardLoadResult).

    Raises:
        ValueError: If loaded layer count does not match stage assignment.
    """
    device = f"{device_type}:{device_index}"

    # Filter
    filtered = filter_state_dict_for_stage(state_dict, stage_assignment)

    # Move to device
    on_device = move_state_dict_to_device(filtered, device)

    # Compute result metadata
    loaded_layers: set[int] = set()
    has_embedding = False
    has_lm_head = False
    has_layer_norm = False

    for key in on_device:
        layer_idx = _extract_layer_index(key)
        if layer_idx is not None:
            loaded_layers.add(layer_idx)
        elif key.startswith(_EMBEDDING_PREFIX):
            has_embedding = True
        elif key.startswith(_LM_HEAD_PREFIX):
            has_lm_head = True
        elif key.startswith(_LAYER_NORM_PREFIX):
            has_layer_norm = True

    result = ShardLoadResult(
        tensor_count=len(on_device),
        layer_count=len(loaded_layers),
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
        has_layer_norm=has_layer_norm,
        device=device,
    )

    # Validate
    validate_shard_load(result, stage_assignment)

    logger.info(
        "Shard loaded for rank %d: %d tensors, %d layers, "
        "embedding=%s, lm_head=%s, device=%s",
        stage_assignment.rank,
        result.tensor_count,
        result.layer_count,
        result.has_embedding,
        result.has_lm_head,
        result.device,
    )

    return on_device, result


def validate_shard_load(
    result: ShardLoadResult,
    stage_assignment: StageAssignment,
) -> None:
    """Validate that the shard load result matches the stage assignment.

    Checks:
      - Layer count matches expected (end_layer - start_layer)
      - Embedding presence matches has_embedding
      - LM head presence matches has_lm_head

    Args:
        result: The shard load result to validate.
        stage_assignment: The expected stage assignment.

    Raises:
        ValueError: If any validation check fails.
    """
    expected_layers = stage_assignment.end_layer - stage_assignment.start_layer

    if result.layer_count != expected_layers:
        raise ValueError(
            f"Shard load validation failed for rank {stage_assignment.rank}: "
            f"expected {expected_layers} layers "
            f"(layers {stage_assignment.start_layer}-{stage_assignment.end_layer}), "
            f"but loaded {result.layer_count}"
        )

    if stage_assignment.has_embedding and not result.has_embedding:
        raise ValueError(
            f"Shard load validation failed for rank {stage_assignment.rank}: "
            f"expected embedding tensors but none were loaded"
        )

    if stage_assignment.has_lm_head and not result.has_lm_head:
        raise ValueError(
            f"Shard load validation failed for rank {stage_assignment.rank}: "
            f"expected LM head tensors but none were loaded"
        )


# Type import for annotations only
import typing

if typing.TYPE_CHECKING:
    import torch
