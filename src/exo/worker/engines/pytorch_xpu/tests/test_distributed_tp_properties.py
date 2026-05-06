# Feature: tensor-parallelism-xpu, Property 6: Rank derivation consistency
"""
Property-based tests for rank derivation consistency.

**Validates: Requirements 9.2**

Uses Hypothesis to generate random instance configs with N nodes and rank
assignments, verifying that:
1. All ranks are unique
2. Ranks form a complete set [0, N)
3. The node with rank 0 has its IP as the master_addr (first lexicographically)
4. The same rank is produced for a given node regardless of which node performs derivation
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"


def _load_distributed() -> types.ModuleType:
    """Load distributed.py directly from file, avoiding __init__.py."""
    module_name = "distributed_isolated"
    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
_derive_rank_assignment = _mod.derive_rank_assignment

# ---------------------------------------------------------------------------
# Hypothesis strategies for generating node IP lists
# ---------------------------------------------------------------------------


@st.composite
def _unique_node_ips(draw: st.DrawFn) -> list[str]:
    """Generate a list of 2–8 unique IPv4 addresses.

    Uses the 10.4.0.x subnet (TB4 subnet) for realism, but the function
    should work with any valid IP strings.
    """
    n = draw(st.integers(min_value=2, max_value=8))
    # Generate unique last octets
    octets = draw(
        st.lists(
            st.integers(min_value=1, max_value=254),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return [f"10.4.0.{octet}" for octet in octets]


@st.composite
def _shuffled_node_ips(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """Generate a list of unique IPs and a shuffled version of the same list.

    Returns (original, shuffled) where both contain the same IPs in
    potentially different orders.
    """
    ips = draw(_unique_node_ips())
    shuffled = draw(st.permutations(ips))
    return ips, list(shuffled)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestRankDerivationConsistencyProperty:
    """Property 6: Rank derivation consistency.

    **Validates: Requirements 9.2**
    """

    @given(node_ips=_unique_node_ips())
    @settings(max_examples=100)
    def test_ranks_are_unique(self, node_ips: list[str]) -> None:
        """For any set of N node IPs, derive_rank_assignment SHALL assign
        exactly one unique rank to each node.

        **Validates: Requirements 9.2**
        """
        assignment = _derive_rank_assignment(node_ips)

        ranks = list(assignment.values())
        assert len(ranks) == len(set(ranks)), (
            f"Ranks are not unique: {assignment}"
        )

    @given(node_ips=_unique_node_ips())
    @settings(max_examples=100)
    def test_ranks_form_complete_set(self, node_ips: list[str]) -> None:
        """For any set of N node IPs, derive_rank_assignment SHALL produce
        ranks that form the complete set [0, N).

        **Validates: Requirements 9.2**
        """
        assignment = _derive_rank_assignment(node_ips)
        n = len(node_ips)

        ranks = sorted(assignment.values())
        expected = list(range(n))
        assert ranks == expected, (
            f"Ranks {ranks} do not form complete set [0, {n}): {assignment}"
        )

    @given(node_ips=_unique_node_ips())
    @settings(max_examples=100)
    def test_rank_zero_maps_to_master_addr(self, node_ips: list[str]) -> None:
        """For any set of N node IPs, the node with rank 0 SHALL have its IP
        as the MASTER_ADDR (the lexicographically first IP).

        **Validates: Requirements 9.2**
        """
        assignment = _derive_rank_assignment(node_ips)

        # Find the node with rank 0
        rank_zero_ip = next(ip for ip, rank in assignment.items() if rank == 0)

        # The master_addr should be the lexicographically smallest IP
        expected_master = sorted(node_ips)[0]
        assert rank_zero_ip == expected_master, (
            f"Rank 0 assigned to {rank_zero_ip}, but MASTER_ADDR should be "
            f"{expected_master} (lexicographically first)"
        )

    @given(data=_shuffled_node_ips())
    @settings(max_examples=100)
    def test_same_rank_regardless_of_derivation_order(
        self, data: tuple[list[str], list[str]]
    ) -> None:
        """For any set of N node IPs, derive_rank_assignment SHALL produce
        the same rank for a given node regardless of the order in which
        the node IPs are provided (simulating different nodes performing
        the derivation with potentially different orderings).

        **Validates: Requirements 9.2**
        """
        original, shuffled = data

        assignment_original = _derive_rank_assignment(original)
        assignment_shuffled = _derive_rank_assignment(shuffled)

        # Both should produce identical mappings
        assert assignment_original == assignment_shuffled, (
            f"Different input orderings produced different assignments:\n"
            f"  Original order {original} → {assignment_original}\n"
            f"  Shuffled order {shuffled} → {assignment_shuffled}"
        )

    @given(node_ips=_unique_node_ips())
    @settings(max_examples=100)
    def test_all_input_nodes_have_assignments(
        self, node_ips: list[str]
    ) -> None:
        """For any set of N node IPs, derive_rank_assignment SHALL return
        a mapping that contains every input node IP as a key.

        **Validates: Requirements 9.2**
        """
        assignment = _derive_rank_assignment(node_ips)

        for ip in node_ips:
            assert ip in assignment, (
                f"Node IP {ip} missing from assignment: {assignment}"
            )
        assert len(assignment) == len(node_ips), (
            f"Assignment has {len(assignment)} entries but expected {len(node_ips)}"
        )
