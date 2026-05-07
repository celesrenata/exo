"""
Unit tests for TB4 topology discovery.

Example-based tests for _classify_topology and select_tb4_interface using
known graph structures.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_TB4_TOPOLOGY_PATH = _THIS_DIR.parent / "tb4_topology.py"


def _load_tb4_topology() -> types.ModuleType:
    """Load tb4_topology.py directly from file, avoiding __init__.py."""
    module_name = "tb4_topology_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _TB4_TOPOLOGY_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_tb4_topology()
_classify_topology = _mod._classify_topology
select_tb4_interface = _mod.select_tb4_interface
TB4Topology = _mod.TB4Topology
TB4Peer = _mod.TB4Peer


# ===========================================================================
# Tests for _classify_topology: Mesh detection
# ===========================================================================


class TestMeshDetection:
    """Test mesh detection with 4 fully-connected nodes.

    **Validates: Requirements 2.2, 2.3**
    """

    def test_4_node_full_mesh(self) -> None:
        """4 nodes where every pair is connected → mesh."""
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C", "D"},
            "B": {"A", "C", "D"},
            "C": {"A", "B", "D"},
            "D": {"A", "B", "C"},
        }
        assert _classify_topology(nodes, graph) == "mesh"

    def test_3_node_full_mesh(self) -> None:
        """3 nodes where every pair is connected → mesh (triangle)."""
        nodes = {"A", "B", "C"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C"},
            "B": {"A", "C"},
            "C": {"A", "B"},
        }
        assert _classify_topology(nodes, graph) == "mesh"

    def test_2_node_full_mesh(self) -> None:
        """2 nodes connected to each other → mesh (simplest complete graph)."""
        nodes = {"A", "B"}
        graph: dict[str, set[str]] = {
            "A": {"B"},
            "B": {"A"},
        }
        assert _classify_topology(nodes, graph) == "mesh"

    def test_5_node_full_mesh(self) -> None:
        """5 nodes where every pair is connected → mesh."""
        nodes = {"n0", "n1", "n2", "n3", "n4"}
        graph: dict[str, set[str]] = {
            node: nodes - {node} for node in nodes
        }
        assert _classify_topology(nodes, graph) == "mesh"


# ===========================================================================
# Tests for _classify_topology: Ring detection
# ===========================================================================


class TestRingDetection:
    """Test ring detection with 4 nodes in a cycle.

    **Validates: Requirements 2.2, 2.4**
    """

    def test_4_node_ring(self) -> None:
        """4 nodes in a cycle: A-B-C-D-A → ring."""
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B", "D"},
            "B": {"A", "C"},
            "C": {"B", "D"},
            "D": {"C", "A"},
        }
        assert _classify_topology(nodes, graph) == "ring"

    def test_5_node_ring(self) -> None:
        """5 nodes in a cycle: A-B-C-D-E-A → ring."""
        nodes = {"A", "B", "C", "D", "E"}
        graph: dict[str, set[str]] = {
            "A": {"B", "E"},
            "B": {"A", "C"},
            "C": {"B", "D"},
            "D": {"C", "E"},
            "E": {"D", "A"},
        }
        assert _classify_topology(nodes, graph) == "ring"

    def test_6_node_ring(self) -> None:
        """6 nodes in a cycle → ring."""
        node_names = [f"n{i}" for i in range(6)]
        nodes = set(node_names)
        graph: dict[str, set[str]] = {node: set() for node in node_names}
        for i in range(6):
            next_idx = (i + 1) % 6
            graph[node_names[i]].add(node_names[next_idx])
            graph[node_names[next_idx]].add(node_names[i])
        assert _classify_topology(nodes, graph) == "ring"

    def test_3_node_ring_is_mesh(self) -> None:
        """3 nodes in a cycle is also a complete graph → classified as mesh.

        A 3-node ring has degree 2 = N-1, making it a complete graph.
        Since mesh is checked first, it's classified as mesh.
        """
        nodes = {"A", "B", "C"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C"},
            "B": {"A", "C"},
            "C": {"A", "B"},
        }
        # This is both a ring and a mesh; mesh takes priority
        assert _classify_topology(nodes, graph) == "mesh"


# ===========================================================================
# Tests for _classify_topology: Partial topology
# ===========================================================================


class TestPartialTopology:
    """Test partial topology with incomplete connections.

    **Validates: Requirements 2.2**
    """

    def test_4_nodes_linear_chain(self) -> None:
        """4 nodes in a line: A-B-C-D (not a ring, not a mesh) → partial."""
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B"},
            "B": {"A", "C"},
            "C": {"B", "D"},
            "D": {"C"},
        }
        assert _classify_topology(nodes, graph) == "partial"

    def test_4_nodes_star_topology(self) -> None:
        """4 nodes in a star: A connected to B, C, D but B/C/D not connected → partial."""
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C", "D"},
            "B": {"A"},
            "C": {"A"},
            "D": {"A"},
        }
        assert _classify_topology(nodes, graph) == "partial"

    def test_4_nodes_one_missing_edge(self) -> None:
        """4 nodes with one edge missing from full mesh → partial."""
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C", "D"},
            "B": {"A", "C", "D"},
            "C": {"A", "B"},  # Missing connection to D
            "D": {"A", "B"},  # Missing connection to C
        }
        assert _classify_topology(nodes, graph) == "partial"

    def test_3_nodes_one_edge_missing(self) -> None:
        """3 nodes with only 2 connected (A-B, A-C, no B-C) → partial."""
        nodes = {"A", "B", "C"}
        graph: dict[str, set[str]] = {
            "A": {"B", "C"},
            "B": {"A"},
            "C": {"A"},
        }
        assert _classify_topology(nodes, graph) == "partial"

    def test_4_nodes_two_disconnected_pairs(self) -> None:
        """4 nodes as two disconnected pairs: A-B and C-D → partial.

        Even though not all nodes can reach each other transitively,
        the function classifies based on the provided reachable_nodes set.
        """
        nodes = {"A", "B", "C", "D"}
        graph: dict[str, set[str]] = {
            "A": {"B"},
            "B": {"A"},
            "C": {"D"},
            "D": {"C"},
        }
        assert _classify_topology(nodes, graph) == "partial"


# ===========================================================================
# Tests for _classify_topology: Unavailable
# ===========================================================================


class TestUnavailable:
    """Test unavailable when fewer than 2 nodes reachable.

    **Validates: Requirements 2.2, 2.6**
    """

    def test_zero_nodes(self) -> None:
        """Empty set of reachable nodes → unavailable."""
        nodes: set[str] = set()
        graph: dict[str, set[str]] = {}
        assert _classify_topology(nodes, graph) == "unavailable"

    def test_single_node(self) -> None:
        """Only 1 node reachable → unavailable."""
        nodes = {"A"}
        graph: dict[str, set[str]] = {"A": set()}
        assert _classify_topology(nodes, graph) == "unavailable"

    def test_single_node_no_graph_entry(self) -> None:
        """Single node with no entry in graph → unavailable."""
        nodes = {"A"}
        graph: dict[str, set[str]] = {}
        assert _classify_topology(nodes, graph) == "unavailable"


# ===========================================================================
# Tests for select_tb4_interface
# ===========================================================================


class TestSelectTB4Interface:
    """Test select_tb4_interface returns correct interface for each topology type.

    **Validates: Requirements 2.1, 2.2**
    """

    def test_unavailable_returns_none(self) -> None:
        """When topology is unavailable, returns None."""
        topology = TB4Topology(
            topology_type="unavailable",
            local_interfaces=[],
            local_ips=[],
            peers=[],
            all_node_ips={},
        )
        assert select_tb4_interface(topology) is None

    def test_unavailable_with_interfaces_returns_none(self) -> None:
        """When topology is unavailable (even with local interfaces), returns None."""
        topology = TB4Topology(
            topology_type="unavailable",
            local_interfaces=["thunderbolt0"],
            local_ips=["10.4.0.1"],
            peers=[],
            all_node_ips={"local": ["10.4.0.1"]},
        )
        assert select_tb4_interface(topology) is None

    def test_mesh_returns_first_interface(self) -> None:
        """For mesh topology, returns the first local TB4 interface."""
        topology = TB4Topology(
            topology_type="mesh",
            local_interfaces=["thunderbolt0", "thunderbolt1", "thunderbolt2"],
            local_ips=["10.4.0.1", "10.4.0.2", "10.4.0.3"],
            peers=[
                TB4Peer(node_ip="10.4.0.4", interface_name="thunderbolt0", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.5", interface_name="thunderbolt1", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.6", interface_name="thunderbolt2", bandwidth_gbps=40.0),
            ],
            all_node_ips={
                "gremlin-1": ["10.4.0.1", "10.4.0.2", "10.4.0.3"],
                "gremlin-2": ["10.4.0.4", "10.4.0.5", "10.4.0.6"],
            },
        )
        result = select_tb4_interface(topology)
        assert result == "thunderbolt0"

    def test_ring_returns_interface_with_most_peers(self) -> None:
        """For ring topology, returns the interface connected to the most peers."""
        topology = TB4Topology(
            topology_type="ring",
            local_interfaces=["thunderbolt0", "thunderbolt1"],
            local_ips=["10.4.0.1", "10.4.0.2"],
            peers=[
                TB4Peer(node_ip="10.4.0.3", interface_name="thunderbolt0", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.4", interface_name="thunderbolt0", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.5", interface_name="thunderbolt1", bandwidth_gbps=40.0),
            ],
            all_node_ips={
                "gremlin-1": ["10.4.0.1", "10.4.0.2"],
                "gremlin-2": ["10.4.0.3"],
                "gremlin-3": ["10.4.0.4"],
                "gremlin-4": ["10.4.0.5"],
            },
        )
        result = select_tb4_interface(topology)
        # thunderbolt0 has 2 peers, thunderbolt1 has 1 peer
        assert result == "thunderbolt0"

    def test_partial_returns_interface_with_most_peers(self) -> None:
        """For partial topology, returns the interface connected to the most peers."""
        topology = TB4Topology(
            topology_type="partial",
            local_interfaces=["thunderbolt0", "thunderbolt1"],
            local_ips=["10.4.0.1", "10.4.0.2"],
            peers=[
                TB4Peer(node_ip="10.4.0.3", interface_name="thunderbolt1", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.4", interface_name="thunderbolt1", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.5", interface_name="thunderbolt1", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.6", interface_name="thunderbolt0", bandwidth_gbps=40.0),
            ],
            all_node_ips={
                "gremlin-1": ["10.4.0.1", "10.4.0.2"],
                "gremlin-2": ["10.4.0.3"],
                "gremlin-3": ["10.4.0.4"],
                "gremlin-4": ["10.4.0.5", "10.4.0.6"],
            },
        )
        result = select_tb4_interface(topology)
        # thunderbolt1 has 3 peers, thunderbolt0 has 1 peer
        assert result == "thunderbolt1"

    def test_no_local_interfaces_returns_none(self) -> None:
        """When topology is available but no local interfaces, returns None."""
        topology = TB4Topology(
            topology_type="mesh",
            local_interfaces=[],
            local_ips=[],
            peers=[
                TB4Peer(node_ip="10.4.0.3", interface_name="thunderbolt0", bandwidth_gbps=40.0),
            ],
            all_node_ips={
                "gremlin-1": ["10.4.0.1"],
                "gremlin-2": ["10.4.0.3"],
            },
        )
        assert select_tb4_interface(topology) is None

    def test_ring_no_peers_falls_back_to_first_interface(self) -> None:
        """For ring/partial with no peers, falls back to first local interface."""
        topology = TB4Topology(
            topology_type="ring",
            local_interfaces=["thunderbolt0", "thunderbolt1"],
            local_ips=["10.4.0.1", "10.4.0.2"],
            peers=[],
            all_node_ips={
                "gremlin-1": ["10.4.0.1", "10.4.0.2"],
                "gremlin-2": ["10.4.0.3"],
            },
        )
        result = select_tb4_interface(topology)
        assert result == "thunderbolt0"

    def test_mesh_single_interface(self) -> None:
        """Mesh with a single local interface returns that interface."""
        topology = TB4Topology(
            topology_type="mesh",
            local_interfaces=["thunderbolt0"],
            local_ips=["10.4.0.1"],
            peers=[
                TB4Peer(node_ip="10.4.0.2", interface_name="thunderbolt0", bandwidth_gbps=40.0),
                TB4Peer(node_ip="10.4.0.3", interface_name="thunderbolt0", bandwidth_gbps=40.0),
            ],
            all_node_ips={
                "gremlin-1": ["10.4.0.1"],
                "gremlin-2": ["10.4.0.2"],
                "gremlin-3": ["10.4.0.3"],
            },
        )
        result = select_tb4_interface(topology)
        assert result == "thunderbolt0"
