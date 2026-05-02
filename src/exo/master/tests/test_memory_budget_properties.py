# Feature: distributed-gpu-sharding, Property 8: Memory budget calculation for heterogeneous nodes
# Feature: distributed-gpu-sharding, Property 9: Memory budget enforcement rejects over-committed placements
"""
Property-based tests for memory budget calculation and enforcement.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

Uses Hypothesis to verify:
- Shared nodes subtract OS overhead from total memory
- Discrete nodes use full VRAM with no overhead
- available_for_model is never negative
- Placement is rejected when shard requirement exceeds budget
- Placement is accepted when shard requirement fits within budget
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.master.memory_budget import (
    MemoryBudget,
    calculate_memory_budget,
    is_placement_within_budget,
)
from exo.shared.types.memory import Memory

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Memory in bytes: 0 to 256 GiB
_memory_bytes = st.integers(min_value=0, max_value=256 * 1024**3)

# OS overhead in bytes: 0 to 16 GiB
_overhead_bytes = st.integers(min_value=0, max_value=16 * 1024**3)

# Architecture type
_architecture = st.sampled_from(["Shared", "Discrete"])


# ---------------------------------------------------------------------------
# Property 8: Memory budget calculation for heterogeneous nodes
# ---------------------------------------------------------------------------


class TestMemoryBudgetCalculationProperty:
    """Property 8: Memory budget calculation for heterogeneous nodes.

    **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
    """

    @given(
        total_bytes=_memory_bytes,
        overhead_bytes=_overhead_bytes,
    )
    @settings(max_examples=100)
    def test_shared_node_subtracts_overhead(
        self,
        total_bytes: int,
        overhead_bytes: int,
    ) -> None:
        """For any Shared memory node, available_for_model SHALL equal
        total_memory - os_overhead, clamped to 0.

        **Validates: Requirements 12.1, 12.2**
        """
        total = Memory.from_bytes(total_bytes)
        budget = calculate_memory_budget(total, "Shared", os_overhead_bytes=overhead_bytes)

        expected_available = max(0, total_bytes - overhead_bytes)

        assert budget.architecture == "Shared"
        assert budget.total_pool.in_bytes == total_bytes
        assert budget.os_overhead.in_bytes == overhead_bytes
        assert budget.available_for_model.in_bytes == expected_available

    @given(
        total_bytes=_memory_bytes,
        overhead_bytes=_overhead_bytes,
    )
    @settings(max_examples=100)
    def test_discrete_node_uses_full_vram(
        self,
        total_bytes: int,
        overhead_bytes: int,
    ) -> None:
        """For any Discrete memory node, available_for_model SHALL equal
        total_memory with no overhead subtraction.

        **Validates: Requirements 12.3**
        """
        total = Memory.from_bytes(total_bytes)
        budget = calculate_memory_budget(total, "Discrete", os_overhead_bytes=overhead_bytes)

        assert budget.architecture == "Discrete"
        assert budget.total_pool.in_bytes == total_bytes
        assert budget.os_overhead.in_bytes == 0
        assert budget.available_for_model.in_bytes == total_bytes

    @given(
        total_bytes=_memory_bytes,
        architecture=_architecture,
        overhead_bytes=_overhead_bytes,
    )
    @settings(max_examples=100)
    def test_available_for_model_never_negative(
        self,
        total_bytes: int,
        architecture: str,
        overhead_bytes: int,
    ) -> None:
        """For any (total_memory, architecture, overhead) tuple,
        available_for_model SHALL never be negative.

        **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
        """
        total = Memory.from_bytes(total_bytes)
        budget = calculate_memory_budget(total, architecture, os_overhead_bytes=overhead_bytes)

        assert budget.available_for_model.in_bytes >= 0

    @given(
        total_bytes=_memory_bytes,
        architecture=_architecture,
        overhead_bytes=_overhead_bytes,
    )
    @settings(max_examples=100)
    def test_architecture_selects_correct_strategy(
        self,
        total_bytes: int,
        architecture: str,
        overhead_bytes: int,
    ) -> None:
        """The placement module SHALL use the memory_architecture field to
        select the correct calculation: Shared subtracts overhead, Discrete
        does not.

        **Validates: Requirements 12.4**
        """
        total = Memory.from_bytes(total_bytes)
        budget = calculate_memory_budget(total, architecture, os_overhead_bytes=overhead_bytes)

        assert budget.architecture == architecture

        if architecture == "Shared":
            assert budget.available_for_model.in_bytes == max(0, total_bytes - overhead_bytes)
            assert budget.os_overhead.in_bytes == overhead_bytes
        else:
            assert budget.available_for_model.in_bytes == total_bytes
            assert budget.os_overhead.in_bytes == 0


# ---------------------------------------------------------------------------
# Property 9: Memory budget enforcement rejects over-committed placements
# ---------------------------------------------------------------------------


class TestMemoryBudgetEnforcementProperty:
    """Property 9: Memory budget enforcement rejects over-committed placements.

    **Validates: Requirements 12.5**
    """

    @given(
        total_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        overhead_bytes=st.integers(min_value=0, max_value=16 * 1024**3),
        shard_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
    )
    @settings(max_examples=100)
    def test_over_committed_shared_placement_rejected(
        self,
        total_bytes: int,
        overhead_bytes: int,
        shard_bytes: int,
    ) -> None:
        """When shard_requirement > available_for_model on a Shared node,
        placement SHALL be rejected.

        **Validates: Requirements 12.5**
        """
        total = Memory.from_bytes(total_bytes)
        budget = calculate_memory_budget(total, "Shared", os_overhead_bytes=overhead_bytes)
        shard_req = Memory.from_bytes(shard_bytes)

        result = is_placement_within_budget(shard_req, budget)

        if shard_bytes > budget.available_for_model.in_bytes:
            assert result is False, (
                f"Placement should be rejected: shard={shard_bytes} > "
                f"available={budget.available_for_model.in_bytes}"
            )
        else:
            assert result is True, (
                f"Placement should be accepted: shard={shard_bytes} <= "
                f"available={budget.available_for_model.in_bytes}"
            )

    @given(
        total_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        shard_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
    )
    @settings(max_examples=100)
    def test_placement_accepted_when_within_budget(
        self,
        total_bytes: int,
        shard_bytes: int,
    ) -> None:
        """When shard_requirement <= available_for_model, placement SHALL
        be accepted regardless of architecture.

        **Validates: Requirements 12.5**
        """
        total = Memory.from_bytes(total_bytes)

        # Test with Discrete (full VRAM available)
        budget = calculate_memory_budget(total, "Discrete")
        shard_req = Memory.from_bytes(shard_bytes)

        result = is_placement_within_budget(shard_req, budget)

        if shard_bytes <= budget.available_for_model.in_bytes:
            assert result is True
        else:
            assert result is False
