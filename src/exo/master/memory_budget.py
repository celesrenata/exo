"""Memory-architecture-aware shard placement budgeting.

Provides memory budget calculations that account for the difference between
shared-memory GPUs (Intel integrated, sharing system RAM) and discrete GPUs
(NVIDIA, with dedicated VRAM).
"""

from __future__ import annotations

from dataclasses import dataclass

from exo.shared.types.memory import Memory


@dataclass(frozen=True)
class MemoryBudget:
    """Calculated memory budget for a node.

    For shared-memory nodes, os_overhead is subtracted from total_pool.
    For discrete GPU nodes, the full VRAM is available (no overhead).
    """

    total_pool: Memory
    os_overhead: Memory
    available_for_model: Memory
    architecture: str  # "Shared" or "Discrete"


def calculate_memory_budget(
    total_memory: Memory,
    architecture: str,
    os_overhead_bytes: int = 2 * 1024**3,  # 2 GiB default
) -> MemoryBudget:
    """Calculate available memory for model placement.

    For Shared memory: available = total_memory - os_overhead (clamped to 0)
    For Discrete memory: available = total_memory (no overhead subtraction)
    """
    overhead = Memory.from_bytes(os_overhead_bytes)

    if architecture == "Shared":
        available_bytes = max(0, total_memory.in_bytes - overhead.in_bytes)
        available = Memory.from_bytes(available_bytes)
    else:
        # Discrete: full VRAM available, no OS overhead
        available = total_memory
        overhead = Memory.from_bytes(0)

    return MemoryBudget(
        total_pool=total_memory,
        os_overhead=overhead,
        available_for_model=available,
        architecture=architecture,
    )


def is_placement_within_budget(
    shard_memory_requirement: Memory,
    budget: MemoryBudget,
) -> bool:
    """Check whether a shard fits within the node's memory budget.

    Returns True if shard_memory_requirement <= budget.available_for_model.
    """
    return shard_memory_requirement <= budget.available_for_model
