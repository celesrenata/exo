# Feature: tensor-parallelism-xpu, Property 1: Topology classification correctness
"""
Property-based tests for TB4 topology classification.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**

Uses Hypothesis to generate random adjacency matrices for 2–8 nodes and verify
that _classify_topology correctly classifies the graph as "mesh", "ring",
"partial", or "unavailable" based on the graph structure definition.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Literal

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_TB4_TOPOLOGY_PATH = _THIS_DIR.parent / "tb4_topology.py"


def _load_tb4_topology() -> types.ModuleType:
    """Load tb4_topology.py directly from file, avoiding __init__.py."""
    module_name = "tb4_topology_isolated"
    spec = importlib.util.spec_from_file_location(module_name, _TB4_TOPOLOGY_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_tb4_topology()
_classify_topology = _mod._classify_topology

# ---------------------------------------------------------------------------
# Helper: determine expected classification from graph structure
# ---------------------------------------------------------------------------


def _expected_classification(
    nodes: list[str],
    adjacency: dict[str, set[str]],
) -> Literal["mesh", "ring", "partial", "unavailable"]:
    """Independently compute the expected topology classification.

    This is a reference implementation that directly encodes the specification
    rules without sharing code with the implementation under test.
    """
    n = len(nodes)

    if n < 2:
        return "unavailable"

    # Check mesh: every node connected to every other node (complete graph)
    is_mesh = all(len(adjacency.get(node, set())) == n - 1 for node in nodes)
    if is_mesh:
        return "mesh"

    # Check ring: each node has exactly 2 neighbors AND forms a single cycle
    if n >= 3:
        all_degree_two = all(
            len(adjacency.get(node, set())) == 2 for node in nodes
        )
        if all_degree_two:
            # Verify single cycle by traversal
            visited: set[str] = set()
            start = nodes[0]
            current = start
            prev: str | None = None

            for _ in range(n):
                visited.add(current)
                neighbors = adjacency.get(current, set())
                next_candidates = neighbors - {prev} if prev else neighbors
                if not next_candidates:
                    break
                prev = current
                current = next(iter(next_candidates))

            if visited == set(nodes) and current == start:
                return "ring"

    return "partial"


# ---------------------------------------------------------------------------
# Hypothesis strategies for graph generation
# ---------------------------------------------------------------------------


def _node_names(n: int) -> list[str]:
    """Generate n node names like 'node_0', 'node_1', etc."""
    return [f"node_{i}" for i in range(n)]


@st.composite
def _random_graph(draw: st.DrawFn) -> tuple[set[str], dict[str, set[str]]]:
    """Generate a random undirected graph with 2–8 nodes.

    Each pair of nodes is independently connected or not (random adjacency matrix).
    The graph is guaranteed to have at least 2 nodes.
    """
    n = draw(st.integers(min_value=2, max_value=8))
    nodes = _node_names(n)

    # Generate random adjacency: for each pair (i, j) where i < j,
    # decide whether they are connected
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}

    for i in range(n):
        for j in range(i + 1, n):
            connected = draw(st.booleans())
            if connected:
                adjacency[nodes[i]].add(nodes[j])
                adjacency[nodes[j]].add(nodes[i])

    return set(nodes), adjacency


@st.composite
def _mesh_graph(draw: st.DrawFn) -> tuple[set[str], dict[str, set[str]]]:
    """Generate a complete graph (mesh) with 2–8 nodes."""
    n = draw(st.integers(min_value=2, max_value=8))
    nodes = _node_names(n)

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for i in range(n):
        for j in range(n):
            if i != j:
                adjacency[nodes[i]].add(nodes[j])

    return set(nodes), adjacency


@st.composite
def _ring_graph(draw: st.DrawFn) -> tuple[set[str], dict[str, set[str]]]:
    """Generate a ring graph with 4–8 nodes.

    Note: N=3 is excluded because a 3-node ring is also a complete graph
    (each node has degree 2 = N-1), so the implementation correctly classifies
    it as "mesh" (mesh is checked first). For N >= 4, a ring (degree 2) is
    distinct from a mesh (degree N-1 >= 3).
    """
    n = draw(st.integers(min_value=4, max_value=8))
    nodes = _node_names(n)

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for i in range(n):
        next_idx = (i + 1) % n
        adjacency[nodes[i]].add(nodes[next_idx])
        adjacency[nodes[next_idx]].add(nodes[i])

    return set(nodes), adjacency


@st.composite
def _unavailable_graph(
    draw: st.DrawFn,
) -> tuple[set[str], dict[str, set[str]]]:
    """Generate a graph with fewer than 2 reachable nodes."""
    choice = draw(st.integers(min_value=0, max_value=1))
    if choice == 0:
        # Empty graph
        return set(), {}
    else:
        # Single node
        nodes = _node_names(1)
        return set(nodes), {nodes[0]: set()}


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestTopologyClassificationProperty:
    """Property 1: Topology classification correctness.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
    """

    @given(graph=_random_graph())
    @settings(max_examples=100)
    def test_classification_matches_graph_structure(
        self,
        graph: tuple[set[str], dict[str, set[str]]],
    ) -> None:
        """For any random adjacency matrix over 2–8 nodes, _classify_topology
        SHALL return the classification that matches the graph structure:
        - "mesh" iff all pairs connected (complete graph)
        - "ring" iff each node has exactly 2 neighbors forming a cycle (N >= 3)
        - "partial" iff at least 2 reachable but neither mesh nor ring
        - "unavailable" iff fewer than 2 reachable

        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
        """
        reachable_nodes, reachability_graph = graph

        result = _classify_topology(reachable_nodes, reachability_graph)
        expected = _expected_classification(
            sorted(reachable_nodes), reachability_graph
        )

        assert result == expected, (
            f"Classification mismatch: got '{result}', expected '{expected}' "
            f"for {len(reachable_nodes)} nodes with adjacency: {reachability_graph}"
        )

    @given(graph=_mesh_graph())
    @settings(max_examples=100)
    def test_mesh_always_classified_as_mesh(
        self,
        graph: tuple[set[str], dict[str, set[str]]],
    ) -> None:
        """For any complete graph (all pairs connected), _classify_topology
        SHALL return "mesh".

        **Validates: Requirements 2.2, 2.3**
        """
        reachable_nodes, reachability_graph = graph

        result = _classify_topology(reachable_nodes, reachability_graph)
        assert result == "mesh", (
            f"Complete graph with {len(reachable_nodes)} nodes "
            f"should be 'mesh', got '{result}'"
        )

    @given(graph=_ring_graph())
    @settings(max_examples=100)
    def test_ring_always_classified_as_ring(
        self,
        graph: tuple[set[str], dict[str, set[str]]],
    ) -> None:
        """For any ring graph (each node has exactly 2 neighbors forming a
        single cycle, N >= 4), _classify_topology SHALL return "ring".

        Note: N=3 rings are also complete graphs and correctly classified as
        "mesh" since mesh is checked first and all pairs are connected.

        **Validates: Requirements 2.2, 2.4**
        """
        reachable_nodes, reachability_graph = graph

        result = _classify_topology(reachable_nodes, reachability_graph)
        assert result == "ring", (
            f"Ring graph with {len(reachable_nodes)} nodes "
            f"should be 'ring', got '{result}'"
        )

    @given(graph=_unavailable_graph())
    @settings(max_examples=100)
    def test_fewer_than_2_nodes_classified_unavailable(
        self,
        graph: tuple[set[str], dict[str, set[str]]],
    ) -> None:
        """For any graph with fewer than 2 reachable nodes, _classify_topology
        SHALL return "unavailable".

        **Validates: Requirements 2.2, 2.6**
        """
        reachable_nodes, reachability_graph = graph

        result = _classify_topology(reachable_nodes, reachability_graph)
        assert result == "unavailable", (
            f"Graph with {len(reachable_nodes)} nodes "
            f"should be 'unavailable', got '{result}'"
        )

    @given(graph=_random_graph())
    @settings(max_examples=100)
    def test_classification_is_exhaustive(
        self,
        graph: tuple[set[str], dict[str, set[str]]],
    ) -> None:
        """For any graph, _classify_topology SHALL return one of the four
        valid classification values.

        **Validates: Requirements 2.2, 2.3, 2.4, 2.6**
        """
        reachable_nodes, reachability_graph = graph

        result = _classify_topology(reachable_nodes, reachability_graph)
        assert result in ("mesh", "ring", "partial", "unavailable"), (
            f"Got unexpected classification '{result}'"
        )
