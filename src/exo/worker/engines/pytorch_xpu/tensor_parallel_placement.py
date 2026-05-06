"""Tensor-Parallel Placement Module.

Validates whether a model can be placed using tensor parallelism and creates
TensorParallelInstance configurations. The placement logic checks:
1. Model card `supports_tensor` field
2. Attention heads divisibility by requested world_size
3. TB4 topology availability
4. Symmetric tensor-parallel group validation

Falls back to pipeline parallelism if tensor parallelism is not feasible.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.4, 8.5
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Literal

from exo.worker.engines.pytorch_xpu.tb4_topology import TB4Topology
from exo.worker.engines.pytorch_xpu.tensor_parallel_instance import (
    TensorParallelInstance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCardInfo:
    """Subset of model card fields relevant to tensor-parallel placement.

    This is a simplified representation of the model card used by the
    placement module. The actual model card may have additional fields.

    Attributes:
        model_id: Model identifier (e.g., "Qwen/Qwen3.5-4B").
        supports_tensor: Whether the model supports tensor parallelism.
        attention_heads: Number of attention heads in the model.
        num_key_value_heads: Number of KV heads (for GQA models).
        intermediate_size: MLP intermediate dimension.
    """

    model_id: str
    supports_tensor: bool
    attention_heads: int
    num_key_value_heads: int = 0
    intermediate_size: int = 0


@dataclass(frozen=True)
class PlacementResult:
    """Result of tensor-parallel placement validation.

    Attributes:
        strategy: The parallelism strategy selected.
        instance: The TensorParallelInstance if strategy is "tensor_parallel".
        reason: Human-readable explanation of the placement decision.
    """

    strategy: Literal["tensor_parallel", "pipeline_parallel"]
    instance: TensorParallelInstance | None = None
    reason: str = ""


def validate_symmetric_groups(group_sizes: list[int]) -> bool:
    """Validate that all tensor-parallel groups have the same size.

    For hybrid parallelism, each tensor-parallel group must have the same
    number of nodes (symmetric tensor parallelism). This ensures uniform
    weight sharding and balanced computation across groups.

    Args:
        group_sizes: List of group sizes (number of nodes in each TP group).

    Returns:
        True if all groups have the same size, False otherwise.
        An empty list returns True (vacuously true).

    Requirements: 8.4

    **Validates: Requirements 8.4**
    """
    if not group_sizes:
        return True
    return all(size == group_sizes[0] for size in group_sizes)


def validate_model_supports_tensor(model_card: ModelCardInfo) -> tuple[bool, str]:
    """Check if a model card indicates tensor parallelism support.

    Args:
        model_card: The model card information to validate.

    Returns:
        Tuple of (is_supported, reason_string).

    Requirements: 7.1, 7.5
    """
    if not model_card.supports_tensor:
        return False, (
            f"Model '{model_card.model_id}' does not support tensor parallelism "
            f"(supports_tensor=False)"
        )
    return True, f"Model '{model_card.model_id}' supports tensor parallelism"


def validate_heads_divisibility(
    model_card: ModelCardInfo,
    world_size: int,
) -> tuple[bool, str]:
    """Verify attention heads are evenly divisible by the requested world_size.

    Args:
        model_card: The model card with attention_heads count.
        world_size: The requested tensor-parallel world size.

    Returns:
        Tuple of (is_valid, reason_string).

    Requirements: 7.2
    """
    if model_card.attention_heads % world_size != 0:
        return False, (
            f"Model '{model_card.model_id}' has {model_card.attention_heads} "
            f"attention heads which is not divisible by world_size={world_size} "
            f"(remainder={model_card.attention_heads % world_size})"
        )

    # Also check KV heads if specified
    if model_card.num_key_value_heads > 0:
        if model_card.num_key_value_heads % world_size != 0:
            return False, (
                f"Model '{model_card.model_id}' has {model_card.num_key_value_heads} "
                f"KV heads which is not divisible by world_size={world_size} "
                f"(remainder={model_card.num_key_value_heads % world_size})"
            )

    # Also check intermediate_size if specified
    if model_card.intermediate_size > 0:
        if model_card.intermediate_size % world_size != 0:
            return False, (
                f"Model '{model_card.model_id}' has intermediate_size="
                f"{model_card.intermediate_size} which is not divisible by "
                f"world_size={world_size} "
                f"(remainder={model_card.intermediate_size % world_size})"
            )

    return True, (
        f"Model '{model_card.model_id}' attention heads ({model_card.attention_heads}) "
        f"are divisible by world_size={world_size}"
    )


def validate_topology_for_tp(
    topology: TB4Topology | None,
    required_world_size: int,
) -> tuple[bool, str]:
    """Check if TB4 topology supports the requested tensor-parallel world size.

    Args:
        topology: The discovered TB4 topology (None if discovery failed).
        required_world_size: Number of nodes needed for tensor parallelism.

    Returns:
        Tuple of (is_feasible, reason_string).

    Requirements: 8.5
    """
    if topology is None:
        return False, "TB4 topology discovery failed (topology is None)"

    if not topology.is_available:
        return False, (
            f"TB4 topology is unavailable (type={topology.topology_type})"
        )

    if topology.world_size < required_world_size:
        return False, (
            f"TB4 topology has {topology.world_size} nodes but "
            f"world_size={required_world_size} is required"
        )

    return True, (
        f"TB4 topology supports world_size={required_world_size} "
        f"(available nodes={topology.world_size}, type={topology.topology_type})"
    )


def place_tensor_parallel(
    model_card: ModelCardInfo,
    topology: TB4Topology | None,
    node_ids: list[str],
    requested_world_size: int | None = None,
    master_port: int = 29500,
) -> PlacementResult:
    """Attempt to place a model using tensor parallelism.

    Validates all preconditions for tensor-parallel placement:
    1. Model supports tensor parallelism (model card check)
    2. Attention heads divisible by world_size
    3. TB4 topology available with sufficient nodes
    4. Symmetric groups (for hybrid mode)

    Falls back to pipeline parallelism if any check fails.

    Args:
        model_card: Model card information for the target model.
        topology: Discovered TB4 topology (None if unavailable).
        node_ids: List of available node IDs for placement.
        requested_world_size: Desired TP world size (defaults to len(node_ids)).
        master_port: Port for the TP process group rendezvous.

    Returns:
        PlacementResult with the selected strategy and instance (if TP).

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.4, 8.5
    """
    world_size = requested_world_size or len(node_ids)

    # Check 1: Model supports tensor parallelism
    supports, reason = validate_model_supports_tensor(model_card)
    if not supports:
        logger.info(f"Falling back to pipeline parallelism: {reason}")
        return PlacementResult(
            strategy="pipeline_parallel",
            reason=reason,
        )

    # Check 2: Attention heads divisible by world_size
    divisible, reason = validate_heads_divisibility(model_card, world_size)
    if not divisible:
        logger.info(f"Falling back to pipeline parallelism: {reason}")
        return PlacementResult(
            strategy="pipeline_parallel",
            reason=reason,
        )

    # Check 3: TB4 topology available
    feasible, reason = validate_topology_for_tp(topology, world_size)
    if not feasible:
        logger.info(f"Falling back to pipeline parallelism: {reason}")
        return PlacementResult(
            strategy="pipeline_parallel",
            reason=reason,
        )

    # Check 4: Symmetric groups (single group for now)
    # With a single TP group, symmetry is trivially satisfied.
    # For hybrid mode, this would validate multiple groups.
    if not validate_symmetric_groups([world_size]):
        reason = "Asymmetric tensor-parallel groups are not supported"
        logger.info(f"Falling back to pipeline parallelism: {reason}")
        return PlacementResult(
            strategy="pipeline_parallel",
            reason=reason,
        )

    # All checks passed — create the TensorParallelInstance
    assert topology is not None  # guaranteed by validate_topology_for_tp

    # Derive rank assignments: sort node_ids for deterministic assignment
    sorted_nodes = sorted(node_ids[:world_size])
    rank_assignments = {node_id: rank for rank, node_id in enumerate(sorted_nodes)}

    # Determine TB4 master address (rank 0's TB4 IP)
    rank_0_node = sorted_nodes[0]
    # Use the first local IP from topology for the master address
    # In production, this would come from the node's TB4 IP assignment
    tb4_master_addr = ""
    if topology.local_ips:
        tb4_master_addr = topology.local_ips[0]
    elif topology.all_node_ips:
        # Use first IP from any node
        for ips in topology.all_node_ips.values():
            if ips:
                tb4_master_addr = ips[0]
                break

    # Assign TB4 interfaces to each node
    tb4_interface_by_node: dict[str, str] = {}
    default_interface = (
        topology.local_interfaces[0] if topology.local_interfaces else "thunderbolt0"
    )
    for node_id in sorted_nodes:
        tb4_interface_by_node[node_id] = default_interface

    instance = TensorParallelInstance(
        instance_id=str(uuid.uuid4()),
        model_id=model_card.model_id,
        tp_world_size=world_size,
        tb4_master_addr=tb4_master_addr,
        tb4_master_port=master_port,
        rank_assignments=rank_assignments,
        tb4_interface_by_node=tb4_interface_by_node,
    )

    logger.info(
        f"Placed model '{model_card.model_id}' with tensor parallelism: "
        f"world_size={world_size}, rank_assignments={rank_assignments}"
    )

    return PlacementResult(
        strategy="tensor_parallel",
        instance=instance,
        reason=(
            f"Model '{model_card.model_id}' placed with tensor parallelism "
            f"(world_size={world_size}, topology={topology.topology_type})"
        ),
    )
