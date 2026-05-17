"""
Pipeline stage balancing: per-layer timing export and distribution recommendation.

Extracts per-layer timing data from the PerformanceRecorder's events,
grouped by mode (prefill vs decode), for use by the stage balancing
recommendation algorithm. Provides an optimal layer distribution
recommendation that minimizes the maximum stage time (pipeline bottleneck).

**Validates: Requirements 6.4, 6.5**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from exo.worker.engines.pytorch_xpu.instrumentation import (
    PerformanceRecorder,
)
from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PipelineLayerDistribution,
)

# ---------------------------------------------------------------------------
# Default fixed rank costs (approximate for Qwen3.5-27B on Intel Arc)
# ---------------------------------------------------------------------------

DEFAULT_FIXED_RANK_COSTS: dict[int, float] = {
    0: 0.0001,  # Embedding lookup: ~0.1ms
    -1: 0.002,  # Final rank: norm + lm_head + sampling: ~2ms
}

# ---------------------------------------------------------------------------
# PerLayerTimingData — per-layer timing extracted from instrumentation
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class PerLayerTimingData:
    """
    Per-layer timing data extracted from PerformanceRecorder events.

    Contains mean timing for both decode and prefill modes, along with
    the count of observations for each mode.
    """

    layer_index: int
    """Global layer index in the model."""

    layer_type: str
    """Layer type: 'full_attention' or 'linear_attention'."""

    decode_mean_seconds: float
    """Mean decode step time for this layer in seconds."""

    decode_count: int
    """Number of decode steps recorded for this layer."""

    prefill_mean_seconds: float
    """Mean prefill time for this layer in seconds."""

    prefill_count: int
    """Number of prefill steps recorded for this layer."""


# ---------------------------------------------------------------------------
# PerStageTimingSummary — per-stage aggregated timing
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class PerStageTimingSummary:
    """
    Per-stage aggregated timing summary.

    Aggregates per-layer timing data into per-stage summaries based on
    the pipeline layer distribution.
    """

    rank: int
    """Pipeline rank for this stage."""

    start_layer: int
    """First layer index assigned to this stage (inclusive)."""

    end_layer: int
    """Last layer index assigned to this stage (exclusive)."""

    decode_total_mean_seconds: float
    """Sum of per-layer decode means for layers in this stage."""

    prefill_total_mean_seconds: float
    """Sum of per-layer prefill means for layers in this stage."""

    layer_count: int
    """Number of layers in this stage."""


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_per_layer_timing(
    recorder: PerformanceRecorder,
) -> list[PerLayerTimingData]:
    """
    Extract per-layer timing from the recorder's events.

    Filters events with event_name == "layer_compute", groups by
    metadata["layer_index"], separates by mode ("decode" vs "prefill"),
    and computes mean duration for each layer and mode.

    Args:
        recorder: The PerformanceRecorder containing recorded events.

    Returns:
        List of PerLayerTimingData sorted by layer_index.
    """
    events = recorder.events

    # Group events by layer_index
    layer_events: dict[int, dict[str, list[float]]] = {}
    layer_types: dict[int, str] = {}

    for event in events:
        if event.event_name != "layer_compute":
            continue

        raw_layer_index: object = event.metadata.get("layer_index")
        if raw_layer_index is None:
            continue

        layer_idx: int = int(str(raw_layer_index))
        mode = event.mode

        if layer_idx not in layer_events:
            layer_events[layer_idx] = {"decode": [], "prefill": []}

        # Record layer_type from metadata
        raw_layer_type: object = event.metadata.get("layer_type")
        if raw_layer_type is not None:
            layer_types[layer_idx] = str(raw_layer_type)

        if mode == "decode":
            layer_events[layer_idx]["decode"].append(event.duration)
        elif mode == "prefill":
            layer_events[layer_idx]["prefill"].append(event.duration)

    # Build results sorted by layer_index
    results: list[PerLayerTimingData] = []
    for layer_idx in sorted(layer_events.keys()):
        decode_durations = layer_events[layer_idx]["decode"]
        prefill_durations = layer_events[layer_idx]["prefill"]

        decode_mean = (
            sum(decode_durations) / len(decode_durations)
            if decode_durations
            else 0.0
        )
        prefill_mean = (
            sum(prefill_durations) / len(prefill_durations)
            if prefill_durations
            else 0.0
        )

        results.append(
            PerLayerTimingData(
                layer_index=layer_idx,
                layer_type=layer_types.get(layer_idx, "unknown"),
                decode_mean_seconds=decode_mean,
                decode_count=len(decode_durations),
                prefill_mean_seconds=prefill_mean,
                prefill_count=len(prefill_durations),
            )
        )

    return results


def export_per_stage_timing(
    per_layer: list[PerLayerTimingData],
    distribution: PipelineLayerDistribution,
) -> list[PerStageTimingSummary]:
    """
    Aggregate per-layer timing into per-stage summaries.

    For each rank in the distribution, sums the decode and prefill mean
    times for all layers assigned to that rank.

    Args:
        per_layer: List of per-layer timing data (from export_per_layer_timing).
        distribution: The pipeline layer distribution defining stage boundaries.

    Returns:
        List of PerStageTimingSummary, one per rank, ordered by rank.
    """
    # Build a lookup from layer_index to timing data
    timing_by_layer: dict[int, PerLayerTimingData] = {
        entry.layer_index: entry for entry in per_layer
    }

    results: list[PerStageTimingSummary] = []

    for rank in range(distribution.rank_count):
        assignment = distribution.get_stage_assignment(rank)
        start_layer = assignment.start_layer
        end_layer = assignment.end_layer

        decode_total = 0.0
        prefill_total = 0.0

        for layer_idx in range(start_layer, end_layer):
            timing = timing_by_layer.get(layer_idx)
            if timing is not None:
                decode_total += timing.decode_mean_seconds
                prefill_total += timing.prefill_mean_seconds

        results.append(
            PerStageTimingSummary(
                rank=rank,
                start_layer=start_layer,
                end_layer=end_layer,
                decode_total_mean_seconds=decode_total,
                prefill_total_mean_seconds=prefill_total,
                layer_count=end_layer - start_layer,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Pipeline layer distribution recommendation algorithm
# ---------------------------------------------------------------------------


def _resolve_fixed_rank_costs(
    fixed_rank_costs: dict[int, float] | None,
    world_size: int,
) -> dict[int, float]:
    """Resolve fixed rank costs, mapping -1 to the last rank.

    Args:
        fixed_rank_costs: Raw fixed costs dict (may contain -1 for last rank).
        world_size: Number of pipeline-parallel ranks.

    Returns:
        Dict mapping concrete rank indices to their fixed costs.
    """
    if fixed_rank_costs is None:
        fixed_rank_costs = DEFAULT_FIXED_RANK_COSTS

    resolved: dict[int, float] = {}
    for rank, cost in fixed_rank_costs.items():
        if rank == -1:
            resolved[world_size - 1] = cost
        else:
            resolved[rank] = cost
    return resolved


def _can_partition(
    layer_costs: list[float],
    world_size: int,
    max_stage_time: float,
    rank_fixed_costs: dict[int, float],
) -> tuple[bool, list[int]]:
    """Check if layers can be partitioned into world_size groups with max cost ≤ max_stage_time.

    Uses a greedy left-to-right scan: assign layers to the current rank
    until adding the next layer would exceed the budget, then move to the
    next rank. Each rank must receive at least 1 layer.

    Args:
        layer_costs: Per-layer costs in order.
        world_size: Number of ranks to partition into.
        max_stage_time: Maximum allowed cost per stage (including fixed costs).
        rank_fixed_costs: Resolved fixed costs per rank.

    Returns:
        Tuple of (feasible, layers_per_rank) where layers_per_rank is the
        greedy assignment if feasible, or a partial assignment if not.
    """
    num_layers = len(layer_costs)
    layers_per_rank: list[int] = []
    layer_idx = 0

    for rank in range(world_size):
        remaining_ranks = world_size - rank
        remaining_layers = num_layers - layer_idx

        # Each remaining rank needs at least 1 layer
        if remaining_layers < remaining_ranks:
            return False, layers_per_rank

        fixed_cost = rank_fixed_costs.get(rank, 0.0)
        budget = max_stage_time - fixed_cost

        # If the fixed cost alone exceeds the budget, infeasible
        if budget < 0.0:
            return False, layers_per_rank

        # Greedily assign layers to this rank
        current_sum = 0.0
        count = 0

        while layer_idx < num_layers:
            # Reserve at least 1 layer for each remaining rank after this one
            layers_left_after = num_layers - layer_idx - 1
            ranks_left_after = world_size - rank - 1

            if current_sum + layer_costs[layer_idx] <= budget:
                # Check we can still satisfy remaining ranks
                if layers_left_after >= ranks_left_after:
                    current_sum += layer_costs[layer_idx]
                    layer_idx += 1
                    count += 1
                else:
                    # Must stop here to leave enough layers for remaining ranks
                    break
            else:
                # Adding this layer would exceed budget
                break

        # Each rank must have at least 1 layer
        if count == 0:
            # Force assign 1 layer even if it exceeds budget (infeasible)
            if layer_idx < num_layers:
                layer_idx += 1
                count = 1
                layers_per_rank.append(count)
                return False, layers_per_rank
            else:
                return False, layers_per_rank

        layers_per_rank.append(count)

    # Check all layers were assigned
    if layer_idx < num_layers:
        return False, layers_per_rank

    return True, layers_per_rank


def recommend_pipeline_layer_distribution(
    *,
    per_layer_timing: list[PerLayerTimingData],
    world_size: int,
    fixed_rank_costs: dict[int, float] | None = None,
    mode: str = "decode",
) -> PipelineLayerDistribution:
    """Recommend an optimal layer distribution minimizing the maximum stage time.

    Uses binary search on the answer (max stage time) combined with a greedy
    feasibility check. The algorithm finds the minimum max stage time such
    that all layers can be partitioned into `world_size` contiguous groups
    where each group's total cost (layer costs + fixed rank cost) does not
    exceed the candidate.

    Args:
        per_layer_timing: Per-layer timing data (from export_per_layer_timing).
        world_size: Number of pipeline-parallel ranks.
        fixed_rank_costs: Fixed costs per rank. Use -1 for the last rank.
            Defaults to DEFAULT_FIXED_RANK_COSTS if None.
        mode: Which timing to use: "decode" or "prefill".

    Returns:
        A validated PipelineLayerDistribution that minimizes the pipeline
        bottleneck (maximum stage time).

    Raises:
        ValueError: If world_size < 1, or per_layer_timing is empty,
            or world_size > number of layers.
    """
    num_layers = len(per_layer_timing)

    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}")
    if num_layers == 0:
        raise ValueError("per_layer_timing must not be empty")
    if world_size > num_layers:
        raise ValueError(
            f"world_size ({world_size}) cannot exceed number of layers "
            f"({num_layers})"
        )

    # Extract layer costs based on mode
    if mode == "decode":
        layer_costs = [t.decode_mean_seconds for t in per_layer_timing]
    elif mode == "prefill":
        layer_costs = [t.prefill_mean_seconds for t in per_layer_timing]
    else:
        raise ValueError(f"mode must be 'decode' or 'prefill', got '{mode}'")

    # Resolve fixed rank costs
    rank_fixed_costs = _resolve_fixed_rank_costs(fixed_rank_costs, world_size)

    # Edge case: world_size == 1, all layers go to single rank
    if world_size == 1:
        return PipelineLayerDistribution(
            layers_per_rank=(num_layers,),
            total_layer_count=num_layers,
            rank_count=1,
        )

    # Binary search on the answer (max stage time)
    # Lower bound: the maximum of (any single layer cost + its rank's fixed cost)
    # and (total cost / world_size)
    total_cost = sum(layer_costs)
    max_fixed = max(rank_fixed_costs.values()) if rank_fixed_costs else 0.0

    # The minimum possible max stage time is at least:
    # - The largest single layer (it must go somewhere)
    # - total_cost / world_size (pigeonhole)
    # - Any single layer + its rank's minimum fixed cost
    lower_bound = max(
        max(layer_costs),
        total_cost / world_size,
    )

    # Upper bound: all layers on one rank + max fixed cost
    upper_bound = total_cost + max_fixed

    # Binary search with precision
    best_assignment: list[int] | None = None
    epsilon = 1e-12  # Precision for binary search convergence

    iterations = 0
    max_iterations = 200  # Prevent infinite loops

    while upper_bound - lower_bound > epsilon and iterations < max_iterations:
        mid = (lower_bound + upper_bound) / 2.0
        feasible, assignment = _can_partition(
            layer_costs, world_size, mid, rank_fixed_costs
        )

        if feasible:
            upper_bound = mid
            best_assignment = assignment
        else:
            lower_bound = mid

        iterations += 1

    # If no feasible assignment found during binary search, do one final check
    # at upper_bound
    if best_assignment is None:
        feasible, assignment = _can_partition(
            layer_costs, world_size, upper_bound, rank_fixed_costs
        )
        if feasible:
            best_assignment = assignment
        else:
            # Fallback: uniform distribution (should not happen with valid inputs)
            base = num_layers // world_size
            remainder = num_layers % world_size
            best_assignment = [
                base + 1 if rank < remainder else base
                for rank in range(world_size)
            ]

    return PipelineLayerDistribution(
        layers_per_rank=tuple(best_assignment),
        total_layer_count=num_layers,
        rank_count=world_size,
    )
