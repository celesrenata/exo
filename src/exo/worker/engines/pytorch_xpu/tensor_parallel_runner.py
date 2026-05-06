"""Tensor-Parallel Runner Integration.

Provides the `connect_tensor_parallel_group()` function that handles the
ConnectToGroup logic for tensor-parallel instances. This function:
1. Discovers TB4 topology
2. Selects the best TB4 interface
3. Initializes the TP process group over TB4
4. Verifies connectivity with a test all-reduce
5. Falls back to ethernet-based TP on failure

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from exo.worker.engines.pytorch_xpu.distributed import (
    TensorParallelGroupConfig,
    init_tensor_parallel_group,
    verify_tensor_parallel_group,
)
from exo.worker.engines.pytorch_xpu.tb4_topology import (
    TB4Topology,
    discover_tb4_topology,
    select_tb4_interface,
)
from exo.worker.engines.pytorch_xpu.tensor_parallel_instance import (
    TensorParallelInstance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectResult:
    """Result of attempting to connect to a tensor-parallel group.

    Attributes:
        status: "connected" on success, "failed" if both TB4 and fallback failed.
        interface_used: The network interface name used for the process group.
        transport: Whether TB4 or ethernet was used.
        topology: The discovered TB4 topology (None if discovery failed).
        error_message: Description of the failure (None on success).
    """

    status: Literal["connected", "failed"]
    interface_used: str | None = None
    transport: Literal["tb4", "ethernet"] | None = None
    topology: TB4Topology | None = None
    error_message: str | None = None


async def connect_tensor_parallel_group(
    instance: TensorParallelInstance,
    local_node_id: str,
    ethernet_interface: str | None = None,
    tb4_subnet: str = "10.4.0.0/24",
    expected_nodes: dict[str, list[str]] | None = None,
    probe_timeout_seconds: float = 2.0,
) -> ConnectResult:
    """Connect this node to the tensor-parallel process group.

    Implements the ConnectToGroup handler logic for tensor-parallel instances:
    1. Discover TB4 topology to find available high-bandwidth links
    2. Select the best TB4 interface via select_tb4_interface()
    3. Initialize the TP process group over TB4 using init_tensor_parallel_group()
    4. Verify connectivity with a test all-reduce via verify_tensor_parallel_group()
    5. On failure: attempt fallback to ethernet-based TP with degraded performance

    Args:
        instance: The TensorParallelInstance configuration from placement.
        local_node_id: This node's ID (used to look up rank assignment).
        ethernet_interface: Fallback ethernet interface name (e.g., "eth0").
            If None, fallback is not attempted.
        tb4_subnet: TB4 subnet for topology discovery.
        expected_nodes: Optional mapping of hostname -> TB4 IPs for discovery.
        probe_timeout_seconds: Timeout for TB4 reachability probes.

    Returns:
        ConnectResult indicating success or failure with details.

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    # Determine this node's rank from the instance configuration
    if local_node_id not in instance.rank_assignments:
        return ConnectResult(
            status="failed",
            error_message=(
                f"Local node '{local_node_id}' not found in rank_assignments: "
                f"{list(instance.rank_assignments.keys())}"
            ),
        )

    rank = instance.rank_assignments[local_node_id]
    world_size = instance.tp_world_size

    logger.info(
        f"Connecting to tensor-parallel group: instance={instance.instance_id}, "
        f"rank={rank}, world_size={world_size}, node={local_node_id}"
    )

    # Step 1: Discover TB4 topology
    topology: TB4Topology | None = None
    try:
        topology = await discover_tb4_topology(
            tb4_subnet=tb4_subnet,
            expected_nodes=expected_nodes,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        logger.info(
            f"TB4 topology discovered: type={topology.topology_type}, "
            f"world_size={topology.world_size}, "
            f"local_interfaces={topology.local_interfaces}"
        )
    except Exception as exc:
        logger.warning(
            f"TB4 topology discovery failed: {exc}. Will attempt fallback.",
            exc_info=True,
        )

    # Step 2: Select TB4 interface
    tb4_interface: str | None = None
    if topology is not None and topology.is_available:
        tb4_interface = select_tb4_interface(topology)
        # Also check if the instance has a pre-assigned interface for this node
        if local_node_id in instance.tb4_interface_by_node:
            tb4_interface = instance.tb4_interface_by_node[local_node_id]

    # Step 3: Attempt TB4-based initialization
    if tb4_interface is not None:
        try:
            config = TensorParallelGroupConfig(
                rank=rank,
                world_size=world_size,
                master_addr=instance.tb4_master_addr,
                master_port=instance.tb4_master_port,
                tb4_interface_name=tb4_interface,
            )
            init_tensor_parallel_group(config)

            # Step 4: Verify with test all-reduce
            if verify_tensor_parallel_group(world_size):
                logger.info(
                    f"Tensor-parallel group connected over TB4: "
                    f"rank={rank}, interface={tb4_interface}"
                )
                return ConnectResult(
                    status="connected",
                    interface_used=tb4_interface,
                    transport="tb4",
                    topology=topology,
                )
            else:
                logger.warning(
                    "TB4 tensor-parallel group verification failed. "
                    "Attempting ethernet fallback."
                )
        except Exception as exc:
            logger.warning(
                f"TB4 tensor-parallel group initialization failed: {exc}. "
                f"Attempting ethernet fallback.",
                exc_info=True,
            )

    # Step 5: Fallback to ethernet-based TP
    if ethernet_interface is not None:
        logger.warning(
            "Falling back to ethernet-based tensor parallelism. "
            "Performance will be degraded (2.5 Gbps vs 40 Gbps TB4)."
        )
        try:
            config = TensorParallelGroupConfig(
                rank=rank,
                world_size=world_size,
                master_addr=instance.tb4_master_addr,
                master_port=instance.tb4_master_port,
                tb4_interface_name=ethernet_interface,
                # Longer timeout for ethernet (higher latency)
                init_timeout_seconds=120,
            )
            init_tensor_parallel_group(config)

            if verify_tensor_parallel_group(world_size):
                logger.warning(
                    f"Tensor-parallel group connected over ETHERNET (degraded): "
                    f"rank={rank}, interface={ethernet_interface}"
                )
                return ConnectResult(
                    status="connected",
                    interface_used=ethernet_interface,
                    transport="ethernet",
                    topology=topology,
                )
            else:
                return ConnectResult(
                    status="failed",
                    topology=topology,
                    error_message=(
                        "Ethernet fallback tensor-parallel group verification failed"
                    ),
                )
        except Exception as exc:
            return ConnectResult(
                status="failed",
                topology=topology,
                error_message=(
                    f"Ethernet fallback tensor-parallel group initialization failed: {exc}"
                ),
            )

    # No fallback available
    return ConnectResult(
        status="failed",
        topology=topology,
        error_message=(
            "TB4 initialization failed and no ethernet fallback interface provided"
        ),
    )
