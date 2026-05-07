"""Pipeline stage assignment logic for dividing model layers across nodes.

Divides a transformer model's layers into consecutive pipeline stages,
one per participating node. Earlier stages receive extra layers when the
division is uneven.

Requirements: 5.1, 5.2, 5.3, 5.5, 5.6, 5.8
"""

from __future__ import annotations

from dataclasses import dataclass

_SUPPORTED_WORLD_SIZES: frozenset[int] = frozenset({2, 3, 4})


@dataclass(frozen=True)
class StageAssignment:
    """Assignment of model layers to a pipeline stage.

    Each stage is assigned a contiguous range of transformer layers
    [start_layer, end_layer) and optionally the embedding layer or
    language model head.
    """

    rank: int
    start_layer: int  # Inclusive
    end_layer: int  # Exclusive
    has_embedding: bool  # True for rank 0
    has_lm_head: bool  # True for last rank


def compute_stage_assignments(
    total_layers: int, world_size: int
) -> list[StageAssignment]:
    """Divide layers evenly across pipeline stages.

    Layers are divided as evenly as possible. If total_layers is not
    evenly divisible by world_size, earlier stages get one extra layer.

    Example: 28 layers across 4 nodes → [7, 7, 7, 7]
    Example: 30 layers across 4 nodes → [8, 8, 7, 7]

    Args:
        total_layers: Total number of transformer layers in the model.
            Must be positive.
        world_size: Number of participating nodes. Must be 2, 3, or 4.

    Returns:
        List of StageAssignment objects, one per rank, ordered by rank.

    Raises:
        ValueError: If total_layers <= 0, world_size not in {2, 3, 4},
            or world_size > total_layers.
    """
    if total_layers <= 0:
        raise ValueError(
            f"total_layers must be positive, got {total_layers}"
        )
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {sorted(_SUPPORTED_WORLD_SIZES)}, "
            f"got {world_size}"
        )
    if world_size > total_layers:
        raise ValueError(
            f"world_size ({world_size}) cannot exceed total_layers ({total_layers})"
        )

    base_layers = total_layers // world_size
    remainder = total_layers % world_size

    assignments: list[StageAssignment] = []
    current_layer = 0

    for rank in range(world_size):
        # Earlier stages get one extra layer when division is uneven
        stage_size = base_layers + (1 if rank < remainder else 0)
        start_layer = current_layer
        end_layer = current_layer + stage_size

        assignment = StageAssignment(
            rank=rank,
            start_layer=start_layer,
            end_layer=end_layer,
            has_embedding=(rank == 0),
            has_lm_head=(rank == world_size - 1),
        )
        assignments.append(assignment)
        current_layer = end_layer

    return assignments
