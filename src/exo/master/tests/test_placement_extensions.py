"""Unit tests for placement module extensions for PyTorch distributed.

Tests:
- get_pytorch_ring_hosts_by_node() returns full-mesh hosts
- ethernet-missing node raises ValueError
- filter_cycles_by_memory() with shared-memory nodes subtracts OS overhead
- place_instance() for PyTorchXPURing uses get_pytorch_ring_hosts_by_node()
"""

from unittest.mock import patch

import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    filter_cycles_by_memory,
    get_pytorch_ring_hosts_by_node,
)
from exo.master.tests.conftest import create_node_memory, create_socket_connection
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import CommandId, PlaceInstance
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    GpuMemoryInfo,
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection, Cycle, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceMeta,
    PyTorchXPURingInstance,
)
from exo.shared.types.worker.shards import Sharding


def _build_full_mesh_topology(
    node_ids: list[NodeId],
    ip_map: dict[NodeId, str],
) -> Topology:
    """Build a fully-connected directed topology with socket connections."""
    topology = Topology()
    for nid in node_ids:
        topology.add_node(nid)
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            dst_ip = ip_map[dst]
            topology.add_connection(
                Connection(
                    source=src,
                    sink=dst,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(address=f"/ip4/{dst_ip}/tcp/8000")
                    ),
                )
            )
    return topology


def _build_node_network(
    node_ids: list[NodeId],
    ip_map: dict[NodeId, str],
    interface_type: str = "ethernet",
) -> dict[NodeId, NodeNetworkInfo]:
    """Build node network info with the given interface type."""
    return {
        nid: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address=ip_map[nid],
                    interface_type=interface_type,
                )
            ]
        )
        for nid in node_ids
    }


class TestGetPytorchRingHostsByNode:
    """Test get_pytorch_ring_hosts_by_node() returns full-mesh hosts."""

    def test_full_mesh_hosts_for_three_nodes(self):
        """All nodes should have entries for all other nodes (not just neighbors)."""
        node_a = NodeId("node-a")
        node_b = NodeId("node-b")
        node_c = NodeId("node-c")
        node_ids = [node_a, node_b, node_c]
        ip_map = {
            node_a: "10.1.1.12",
            node_b: "10.1.1.13",
            node_c: "10.1.1.14",
        }

        topology = _build_full_mesh_topology(node_ids, ip_map)
        node_network = _build_node_network(node_ids, ip_map)
        cycle = Cycle(node_ids=node_ids)

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=50000,
            node_network=node_network,
        )

        # Every node has an entry
        assert set(hosts_by_node.keys()) == {node_a, node_b, node_c}

        # Each node has 3 hosts (one per node in cycle)
        for nid in node_ids:
            assert len(hosts_by_node[nid]) == 3

        # Node A's view: self=0.0.0.0, B=10.1.1.13, C=10.1.1.14
        assert hosts_by_node[node_a][0] == Host(ip="0.0.0.0", port=50000)
        assert hosts_by_node[node_a][1] == Host(ip="10.1.1.13", port=50000)
        assert hosts_by_node[node_a][2] == Host(ip="10.1.1.14", port=50000)

        # Node B's view: A=10.1.1.12, self=0.0.0.0, C=10.1.1.14
        assert hosts_by_node[node_b][0] == Host(ip="10.1.1.12", port=50000)
        assert hosts_by_node[node_b][1] == Host(ip="0.0.0.0", port=50000)
        assert hosts_by_node[node_b][2] == Host(ip="10.1.1.14", port=50000)

        # Node C's view: A=10.1.1.12, B=10.1.1.13, self=0.0.0.0
        assert hosts_by_node[node_c][0] == Host(ip="10.1.1.12", port=50000)
        assert hosts_by_node[node_c][1] == Host(ip="10.1.1.13", port=50000)
        assert hosts_by_node[node_c][2] == Host(ip="0.0.0.0", port=50000)

    def test_no_placeholder_ips(self):
        """Unlike MLX ring, PyTorch ring should never use placeholder IPs."""
        node_a = NodeId("node-a")
        node_b = NodeId("node-b")
        node_c = NodeId("node-c")
        node_d = NodeId("node-d")
        node_ids = [node_a, node_b, node_c, node_d]
        ip_map = {
            node_a: "10.1.1.12",
            node_b: "10.1.1.13",
            node_c: "10.1.1.14",
            node_d: "10.1.1.15",
        }

        topology = _build_full_mesh_topology(node_ids, ip_map)
        node_network = _build_node_network(node_ids, ip_map)
        cycle = Cycle(node_ids=node_ids)

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=50000,
            node_network=node_network,
        )

        for nid in node_ids:
            for host in hosts_by_node[nid]:
                # No RFC 5737 TEST-NET-2 placeholder
                assert host.ip != "198.51.100.1", (
                    f"Found placeholder IP for node {nid}"
                )

    def test_empty_cycle_returns_empty(self):
        """Empty cycle should return empty dict."""
        cycle = Cycle(node_ids=[])
        topology = Topology()
        result = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=50000,
            node_network={},
        )
        assert result == {}


class TestEthernetMissingNodeRaisesValueError:
    """Test that missing ethernet connectivity raises ValueError."""

    def test_missing_connection_raises(self):
        """Node without any connection to another node should raise ValueError."""
        node_a = NodeId("node-a")
        node_b = NodeId("node-b")
        node_ids = [node_a, node_b]

        # Build topology with only one direction (A -> B, but not B -> A)
        topology = Topology()
        topology.add_node(node_a)
        topology.add_node(node_b)
        topology.add_connection(
            Connection(
                source=node_a,
                sink=node_b,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(address="/ip4/10.1.1.13/tcp/8000")
                ),
            )
        )

        node_network = {
            node_a: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="eth0", ip_address="10.1.1.12", interface_type="ethernet"
                    )
                ]
            ),
            node_b: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="eth0", ip_address="10.1.1.13", interface_type="ethernet"
                    )
                ]
            ),
        }

        cycle = Cycle(node_ids=node_ids)

        with pytest.raises(ValueError, match="ethernet connectivity"):
            get_pytorch_ring_hosts_by_node(
                selected_cycle=cycle,
                cycle_digraph=topology,
                ephemeral_port=50000,
                node_network=node_network,
            )


class TestFilterCyclesByMemoryWithGpuInfo:
    """Test filter_cycles_by_memory() with shared-memory nodes subtracts OS overhead."""

    def test_shared_memory_subtracts_overhead(self):
        """Shared-memory nodes should have OS overhead subtracted from available memory."""
        node_a = NodeId("node-a")
        node_b = NodeId("node-b")

        # Node A: Shared memory, 8 GiB available RAM
        # After 2 GiB overhead: 6 GiB available for model
        mem_a = MemoryUsage.from_bytes(
            ram_total=8 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=0,
            swap_available=0,
        )
        mem_a.gpu_info = GpuMemoryInfo(
            device_type="xpu",
            memory_architecture="Shared",
            gpu_total_memory=Memory.from_bytes(8 * 1024**3),
            gpu_available_memory=Memory.from_bytes(8 * 1024**3),
        )

        # Node B: No GPU info, 4 GiB available (backward compat)
        mem_b = MemoryUsage.from_bytes(
            ram_total=4 * 1024**3,
            ram_available=4 * 1024**3,
            swap_total=0,
            swap_available=0,
        )

        node_memory = {node_a: mem_a, node_b: mem_b}

        topology = Topology()
        topology.add_node(node_a)
        topology.add_node(node_b)
        topology.add_connection(
            Connection(
                source=node_a, sink=node_b, edge=create_socket_connection(1)
            )
        )
        topology.add_connection(
            Connection(
                source=node_b, sink=node_a, edge=create_socket_connection(2)
            )
        )

        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        assert len(cycles) == 1

        # Total available: 6 GiB (shared after overhead) + 4 GiB (no GPU) = 10 GiB
        # Should pass for 10 GiB requirement
        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(10 * 1024**3)
        )
        assert len(result) == 1

        # Should fail for 11 GiB requirement
        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(11 * 1024**3)
        )
        assert len(result) == 0

    def test_discrete_gpu_uses_vram(self):
        """Discrete GPU nodes should use GPU VRAM, not system RAM."""
        node_a = NodeId("node-a")

        # Node A: Discrete GPU with 12 GiB VRAM, 32 GiB system RAM
        mem_a = MemoryUsage.from_bytes(
            ram_total=32 * 1024**3,
            ram_available=32 * 1024**3,
            swap_total=0,
            swap_available=0,
        )
        mem_a.gpu_info = GpuMemoryInfo(
            device_type="cuda",
            memory_architecture="Discrete",
            gpu_total_memory=Memory.from_bytes(12 * 1024**3),
            gpu_available_memory=Memory.from_bytes(12 * 1024**3),
        )

        node_memory = {node_a: mem_a}

        topology = Topology()
        topology.add_node(node_a)
        cycles = [Cycle(node_ids=[node_a])]

        # Should use 12 GiB VRAM, not 32 GiB system RAM
        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(12 * 1024**3)
        )
        assert len(result) == 1

        # 13 GiB exceeds 12 GiB VRAM
        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(13 * 1024**3)
        )
        assert len(result) == 0

    def test_backward_compat_no_gpu_info(self):
        """Nodes without gpu_info should use ram_available (backward compat)."""
        node_a = NodeId("node-a")

        mem_a = MemoryUsage.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=10 * 1024**3,
            swap_total=0,
            swap_available=0,
        )
        # No gpu_info set

        node_memory = {node_a: mem_a}
        cycles = [Cycle(node_ids=[node_a])]

        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(10 * 1024**3)
        )
        assert len(result) == 1

        result = filter_cycles_by_memory(
            cycles, node_memory, Memory.from_bytes(11 * 1024**3)
        )
        assert len(result) == 0


class TestPlaceInstancePyTorchXPURing:
    """Test place_instance() for PyTorchXPURing uses get_pytorch_ring_hosts_by_node()."""

    def test_pytorch_xpu_ring_uses_full_mesh_hosts(self):
        """PyTorchXPURing placement should produce full-mesh hosts (not just neighbors)."""
        node_a = NodeId("node-a")
        node_b = NodeId("node-b")
        node_c = NodeId("node-c")
        node_ids = [node_a, node_b, node_c]
        ip_map = {
            node_a: "10.1.1.12",
            node_b: "10.1.1.13",
            node_c: "10.1.1.14",
        }

        topology = _build_full_mesh_topology(node_ids, ip_map)
        node_network = _build_node_network(node_ids, ip_map)
        node_memory = {
            nid: create_node_memory(10 * 1024 * 1024)  # 10 GiB
            for nid in node_ids
        }

        model_card = ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(1000),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )

        command = PlaceInstance(
            command_id=CommandId(),
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.PyTorchXPURing,
            min_nodes=1,
        )

        placements = place_instance(
            command, topology, {}, node_memory, node_network
        )

        assert len(placements) == 1
        instance = list(placements.values())[0]
        assert isinstance(instance, PyTorchXPURingInstance)

        # Verify full-mesh: every node's host list should have no placeholder IPs
        for nid, hosts in instance.hosts_by_node.items():
            assert len(hosts) == len(instance.shard_assignments.node_to_runner)
            for host in hosts:
                assert host.ip != "198.51.100.1", (
                    f"Found placeholder IP in PyTorchXPURing hosts for node {nid}"
                )
