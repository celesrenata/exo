"""Property-based tests for pipeline stage assignment logic.

Uses Hypothesis to verify that compute_stage_assignments() produces valid
partitionings for any valid combination of total_layers and world_size.

**Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 5.8**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.pipeline.stage import (
    StageAssignment,
    compute_stage_assignments,
)


# --- Strategies ---

# world_size sampled from the supported set {2, 3, 4}
world_size_st = st.sampled_from([2, 3, 4])

# Strategy that generates (total_layers, world_size) pairs where
# total_layers >= world_size (required precondition)
valid_inputs_st = world_size_st.flatmap(
    lambda ws: st.tuples(
        st.integers(min_value=ws, max_value=100),
        st.just(ws),
    )
)


class TestPipelineStagePartitioningProperty:
    """Property 4: Pipeline stage assignment produces valid partitioning.

    *For any* total_layers > 0 and world_size in {2, 3, 4} where
    world_size <= total_layers, the stage assignment function SHALL produce
    exactly world_size stages such that:
    (a) the stages cover all layers exactly once with no gaps or overlaps,
    (b) the layer count per stage differs by at most 1 between any two stages,
    (c) rank 0 has has_embedding=True,
    (d) rank world_size-1 has has_lm_head=True, and
    (e) each rank maps to exactly one contiguous range of layers.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6, 5.8**
    """

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_stages_cover_all_layers_no_gaps_no_overlaps(
        self, inputs: tuple[int, int]
    ) -> None:
        """(a) Stages cover all layers exactly once with no gaps or overlaps.

        Requirement 5.1: THE Pipeline_Parallelism SHALL divide a transformer
        model's layers into N consecutive Pipeline_Stages.
        Requirement 5.2: THE Pipeline_Parallelism SHALL assign each
        Pipeline_Stage to exactly one node in the cluster.
        """
        total_layers, world_size = inputs
        assignments = compute_stage_assignments(total_layers, world_size)

        # Exactly world_size stages produced
        assert len(assignments) == world_size

        # First stage starts at layer 0
        assert assignments[0].start_layer == 0

        # Last stage ends at total_layers
        assert assignments[-1].end_layer == total_layers

        # No gaps: each stage starts where the previous one ended
        for i in range(len(assignments) - 1):
            assert assignments[i].end_layer == assignments[i + 1].start_layer, (
                f"Gap between stage {i} (end={assignments[i].end_layer}) "
                f"and stage {i+1} (start={assignments[i+1].start_layer})"
            )

        # Total layers covered equals total_layers (no overlaps)
        total_covered = sum(
            a.end_layer - a.start_layer for a in assignments
        )
        assert total_covered == total_layers

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_layer_count_differs_by_at_most_one(
        self, inputs: tuple[int, int]
    ) -> None:
        """(b) Layer count per stage differs by at most 1 between any two stages.

        Requirement 5.3: WHEN Qwen3_5_4B is loaded across 4 nodes, THE
        Pipeline_Parallelism SHALL assign approximately equal numbers of
        layers to each node.
        """
        total_layers, world_size = inputs
        assignments = compute_stage_assignments(total_layers, world_size)

        sizes = [a.end_layer - a.start_layer for a in assignments]
        min_size = min(sizes)
        max_size = max(sizes)

        assert max_size - min_size <= 1, (
            f"Layer counts differ by more than 1: sizes={sizes}, "
            f"min={min_size}, max={max_size}"
        )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_rank_0_has_embedding(
        self, inputs: tuple[int, int]
    ) -> None:
        """(c) Rank 0 has has_embedding=True.

        Requirement 5.5: THE Pipeline_Parallelism SHALL execute the embedding
        layer and first Pipeline_Stage on the first node in the pipeline.
        """
        total_layers, world_size = inputs
        assignments = compute_stage_assignments(total_layers, world_size)

        # Rank 0 must have embedding
        assert assignments[0].has_embedding is True

        # No other rank has embedding
        for assignment in assignments[1:]:
            assert assignment.has_embedding is False, (
                f"Rank {assignment.rank} should not have has_embedding=True"
            )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_last_rank_has_lm_head(
        self, inputs: tuple[int, int]
    ) -> None:
        """(d) Rank world_size-1 has has_lm_head=True.

        Requirement 5.6: THE Pipeline_Parallelism SHALL execute the final
        Pipeline_Stage and language model head on the last node in the pipeline.
        """
        total_layers, world_size = inputs
        assignments = compute_stage_assignments(total_layers, world_size)

        # Last rank must have lm_head
        assert assignments[-1].has_lm_head is True
        assert assignments[-1].rank == world_size - 1

        # No other rank has lm_head
        for assignment in assignments[:-1]:
            assert assignment.has_lm_head is False, (
                f"Rank {assignment.rank} should not have has_lm_head=True"
            )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_each_rank_maps_to_contiguous_range(
        self, inputs: tuple[int, int]
    ) -> None:
        """(e) Each rank maps to exactly one contiguous range of layers.

        Requirement 5.8: THE Pipeline_Parallelism SHALL support variable
        numbers of nodes (2, 3, or 4) for the pipeline without code changes.

        Each stage has start_layer < end_layer (non-empty) and ranks are
        sequential from 0 to world_size-1.
        """
        total_layers, world_size = inputs
        assignments = compute_stage_assignments(total_layers, world_size)

        for i, assignment in enumerate(assignments):
            # Rank matches position
            assert assignment.rank == i, (
                f"Expected rank {i}, got {assignment.rank}"
            )

            # Each stage has at least one layer (contiguous non-empty range)
            assert assignment.start_layer < assignment.end_layer, (
                f"Rank {i} has empty range: "
                f"[{assignment.start_layer}, {assignment.end_layer})"
            )

            # start_layer and end_layer are non-negative
            assert assignment.start_layer >= 0
            assert assignment.end_layer >= 0
