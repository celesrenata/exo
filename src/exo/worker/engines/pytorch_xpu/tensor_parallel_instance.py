"""Tensor-Parallel Instance Configuration.

Defines the TensorParallelInstance dataclass that represents a placed
tensor-parallel inference instance. This is the shared type used by the
placement module, runner, and master to coordinate tensor-parallel
model execution across multiple nodes connected via Thunderbolt 4.

Requirements: 8.1, 8.2, 8.3
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParallelInstance:
    """Instance configuration for tensor-parallel inference.

    Created by the placement module when a model supports tensor parallelism
    and TB4 topology is available. Contains all information needed by each
    runner to initialize its tensor-parallel process group and begin inference.

    Unlike pipeline-parallel instances (which assign layer ranges to nodes),
    tensor-parallel instances assign ALL layers to ALL nodes but with sharded
    weights. Each node holds 1/tp_world_size of the weight matrices.

    Attributes:
        instance_id: Unique identifier for this inference instance.
        model_id: The model being served (e.g., "Qwen/Qwen3.5-4B").
        tp_world_size: Number of tensor-parallel ranks (typically 4 for full cluster).
        tb4_master_addr: TB4 IP address of rank 0 (used for env:// rendezvous).
        tb4_master_port: Ephemeral port for the TP process group.
        rank_assignments: Mapping of node_id -> TP rank (0-indexed).
        tb4_interface_by_node: Mapping of node_id -> TB4 interface name for
            GLOO_SOCKET_IFNAME on each node.
        pipeline_group_id: Optional group ID for hybrid TP+PP mode.
        pipeline_rank: This group's rank in the pipeline (hybrid mode).
        pipeline_world_size: Total pipeline stages (hybrid mode).

    Requirements: 8.1, 8.2, 8.3
    """

    instance_id: str
    model_id: str
    tp_world_size: int
    tb4_master_addr: str
    tb4_master_port: int
    rank_assignments: dict[str, int]  # node_id -> TP rank
    tb4_interface_by_node: dict[str, str]  # node_id -> TB4 interface name
    # Optional pipeline parallelism fields for hybrid mode:
    pipeline_group_id: str | None = None
    pipeline_rank: int | None = None
    pipeline_world_size: int | None = None
