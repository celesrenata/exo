# Feature: pipeline-parallelism-optimization, Property 1: Layer assignment produces complete, non-overlapping coverage
"""
Property-based tests for pipeline-parallel layer assignment.

**Validates: Requirements 1.1, 1.3**

Uses Hypothesis to generate arbitrary n_layers (1..128) and world_size (1..n_layers),
then verifies that compute_layer_assignment produces complete, non-overlapping coverage
of all layers across all ranks.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_SHARD_PATH = _THIS_DIR.parent / "pipeline_parallel_shard.py"


def _load_pipeline_parallel_shard() -> types.ModuleType:
    """Load pipeline_parallel_shard.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_parallel_shard_isolated"
    spec = importlib.util.spec_from_file_location(module_name, _PIPELINE_SHARD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_parallel_shard()
compute_layer_assignment = _mod.compute_layer_assignment
PipelineStageConfig = _mod.PipelineStageConfig


# ---------------------------------------------------------------------------
# Property 1: Layer assignment produces complete, non-overlapping coverage
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=1, max_value=128),
    data=st.data(),
)
def test_layer_assignment_complete_coverage(n_layers: int, data: st.DataObject) -> None:
    """Union of all rank ranges equals [0, n_layers) with no gaps or overlaps.

    **Validates: Requirements 1.1, 1.3**
    """
    world_size = data.draw(st.integers(min_value=1, max_value=n_layers), label="world_size")

    # Collect all layer assignments
    all_layers: set[int] = set()
    expected_layers = set(range(n_layers))

    for rank in range(world_size):
        start, end = compute_layer_assignment(n_layers, world_size, rank)

        # Each rank's range should not overlap with previously seen layers
        rank_layers = set(range(start, end))
        overlap = all_layers & rank_layers
        assert overlap == set(), (
            f"Overlap detected at rank {rank}: layers {overlap} already assigned. "
            f"n_layers={n_layers}, world_size={world_size}"
        )
        all_layers |= rank_layers

    # Union of all rank ranges must equal [0, n_layers)
    assert all_layers == expected_layers, (
        f"Coverage mismatch: missing={expected_layers - all_layers}, "
        f"extra={all_layers - expected_layers}. "
        f"n_layers={n_layers}, world_size={world_size}"
    )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=1, max_value=128),
    data=st.data(),
)
def test_layer_assignment_max_layers_per_rank(n_layers: int, data: st.DataObject) -> None:
    """Each rank gets at most ceil(n_layers / world_size) layers.

    **Validates: Requirements 1.1, 1.3**
    """
    world_size = data.draw(st.integers(min_value=1, max_value=n_layers), label="world_size")
    max_allowed = math.ceil(n_layers / world_size)

    for rank in range(world_size):
        start, end = compute_layer_assignment(n_layers, world_size, rank)
        num_layers_for_rank = end - start
        assert num_layers_for_rank <= max_allowed, (
            f"Rank {rank} got {num_layers_for_rank} layers, "
            f"but max allowed is {max_allowed}. "
            f"n_layers={n_layers}, world_size={world_size}"
        )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=1, max_value=128),
    data=st.data(),
)
def test_layer_assignment_start_less_than_end(n_layers: int, data: st.DataObject) -> None:
    """start_layer < end_layer for all ranks when world_size <= n_layers.

    Every rank gets at least 1 layer when world_size <= n_layers.

    **Validates: Requirements 1.1, 1.3**
    """
    world_size = data.draw(st.integers(min_value=1, max_value=n_layers), label="world_size")

    for rank in range(world_size):
        start, end = compute_layer_assignment(n_layers, world_size, rank)
        assert start < end, (
            f"Rank {rank}: start_layer ({start}) >= end_layer ({end}). "
            f"n_layers={n_layers}, world_size={world_size}"
        )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_layer_assignment_single_layer_single_rank() -> None:
    """Edge case: n_layers=1, world_size=1."""
    start, end = compute_layer_assignment(1, 1, 0)
    assert start == 0
    assert end == 1


def test_layer_assignment_n_layers_equals_world_size() -> None:
    """Edge case: n_layers == world_size (each rank gets exactly 1 layer)."""
    n_layers = 8
    world_size = 8
    for rank in range(world_size):
        start, end = compute_layer_assignment(n_layers, world_size, rank)
        assert end - start == 1, f"Rank {rank} should get exactly 1 layer"
        assert start == rank


def test_layer_assignment_32_layers_4_ranks() -> None:
    """Concrete example: 32-layer model on 4 nodes (Qwen3.5-4B on gremlin cluster)."""
    assignments = [compute_layer_assignment(32, 4, r) for r in range(4)]
    assert assignments == [(0, 8), (8, 16), (16, 24), (24, 32)]


def test_layer_assignment_not_evenly_divisible() -> None:
    """When n_layers is not evenly divisible, earlier ranks get one extra layer."""
    # 10 layers, 3 ranks: base=3, remainder=1
    # rank 0 gets 4 layers (base+1), ranks 1-2 get 3 layers each
    # rank 0: [0, 4), rank 1: [4, 7), rank 2: [7, 10)
    assignments = [compute_layer_assignment(10, 3, r) for r in range(3)]
    assert assignments == [(0, 4), (4, 7), (7, 10)]


def test_layer_assignment_invalid_inputs() -> None:
    """Validation: invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        compute_layer_assignment(0, 1, 0)  # n_layers < 1

    with pytest.raises(ValueError):
        compute_layer_assignment(10, 0, 0)  # world_size < 1

    with pytest.raises(ValueError):
        compute_layer_assignment(10, 4, -1)  # rank < 0

    with pytest.raises(ValueError):
        compute_layer_assignment(10, 4, 4)  # rank >= world_size


# ---------------------------------------------------------------------------
# Property 3: Communication count equals world_size minus one
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=2, max_value=128),
    data=st.data(),
)
def test_communication_count_equals_world_size_minus_one(
    n_layers: int, data: st.DataObject
) -> None:
    """For world_size >= 2, adjacent stage boundaries equal world_size - 1.

    Each adjacent pair (rank, rank+1) where end_layer[rank] == start_layer[rank+1]
    represents one point-to-point send/recv communication. A full pipeline pass
    requires exactly world_size - 1 such communications.

    **Validates: Requirements 3.4**
    """
    world_size = data.draw(st.integers(min_value=2, max_value=n_layers), label="world_size")

    # Compute all assignments
    assignments = [
        compute_layer_assignment(n_layers, world_size, rank)
        for rank in range(world_size)
    ]

    # Count adjacent stage pairs where end_layer[rank] == start_layer[rank+1]
    communication_count = 0
    for rank in range(world_size - 1):
        _, end_layer = assignments[rank]
        next_start, _ = assignments[rank + 1]
        if end_layer == next_start:
            communication_count += 1

    assert communication_count == world_size - 1, (
        f"Expected {world_size - 1} communications, got {communication_count}. "
        f"n_layers={n_layers}, world_size={world_size}, assignments={assignments}"
    )
