"""Unit tests for memory budget calculator.

Tests specific examples and edge cases for memory budget calculation
and placement enforcement.

Requirements: 12.1, 12.2, 12.3
"""

from __future__ import annotations

from exo.master.memory_budget import (
    calculate_memory_budget,
    is_placement_within_budget,
)
from exo.shared.types.memory import Memory


class TestCalculateMemoryBudget:
    def test_shared_node_32gib_ram(self) -> None:
        """Shared node with 32 GiB RAM → available = 30 GiB (32 - 2 overhead)."""
        total = Memory.from_bytes(32 * 1024**3)
        budget = calculate_memory_budget(total, "Shared")

        assert budget.total_pool.in_bytes == 32 * 1024**3
        assert budget.os_overhead.in_bytes == 2 * 1024**3
        assert budget.available_for_model.in_bytes == 30 * 1024**3
        assert budget.architecture == "Shared"

    def test_discrete_node_12gib_vram(self) -> None:
        """Discrete node with 12 GiB VRAM → available = 12 GiB (no overhead)."""
        total = Memory.from_bytes(12 * 1024**3)
        budget = calculate_memory_budget(total, "Discrete")

        assert budget.total_pool.in_bytes == 12 * 1024**3
        assert budget.os_overhead.in_bytes == 0
        assert budget.available_for_model.in_bytes == 12 * 1024**3
        assert budget.architecture == "Discrete"

    def test_custom_overhead_value(self) -> None:
        """Custom overhead value is respected."""
        total = Memory.from_bytes(16 * 1024**3)
        custom_overhead = 4 * 1024**3  # 4 GiB
        budget = calculate_memory_budget(total, "Shared", os_overhead_bytes=custom_overhead)

        assert budget.os_overhead.in_bytes == custom_overhead
        assert budget.available_for_model.in_bytes == 12 * 1024**3

    def test_zero_total_memory(self) -> None:
        """Zero total memory → available = 0 (no negative values)."""
        total = Memory.from_bytes(0)
        budget = calculate_memory_budget(total, "Shared")

        assert budget.total_pool.in_bytes == 0
        assert budget.available_for_model.in_bytes == 0

    def test_overhead_exceeds_total_clamps_to_zero(self) -> None:
        """When overhead exceeds total, available clamps to 0."""
        total = Memory.from_bytes(1 * 1024**3)  # 1 GiB
        budget = calculate_memory_budget(total, "Shared", os_overhead_bytes=4 * 1024**3)

        assert budget.available_for_model.in_bytes == 0

    def test_discrete_ignores_overhead_parameter(self) -> None:
        """Discrete architecture ignores the overhead parameter entirely."""
        total = Memory.from_bytes(8 * 1024**3)
        budget = calculate_memory_budget(total, "Discrete", os_overhead_bytes=99 * 1024**3)

        assert budget.os_overhead.in_bytes == 0
        assert budget.available_for_model.in_bytes == 8 * 1024**3


class TestIsPlacementWithinBudget:
    def test_shard_fits_within_budget(self) -> None:
        """Shard that fits within budget is accepted."""
        budget = calculate_memory_budget(Memory.from_bytes(32 * 1024**3), "Shared")
        shard_req = Memory.from_bytes(20 * 1024**3)

        assert is_placement_within_budget(shard_req, budget) is True

    def test_shard_exceeds_budget(self) -> None:
        """Shard that exceeds budget is rejected."""
        budget = calculate_memory_budget(Memory.from_bytes(32 * 1024**3), "Shared")
        shard_req = Memory.from_bytes(31 * 1024**3)  # > 30 GiB available

        assert is_placement_within_budget(shard_req, budget) is False

    def test_shard_exactly_equals_budget(self) -> None:
        """Shard that exactly equals available memory is accepted."""
        budget = calculate_memory_budget(Memory.from_bytes(32 * 1024**3), "Shared")
        shard_req = Memory.from_bytes(30 * 1024**3)  # exactly 30 GiB

        assert is_placement_within_budget(shard_req, budget) is True

    def test_zero_shard_always_accepted(self) -> None:
        """Zero-size shard is always accepted."""
        budget = calculate_memory_budget(Memory.from_bytes(0), "Shared")
        shard_req = Memory.from_bytes(0)

        assert is_placement_within_budget(shard_req, budget) is True
