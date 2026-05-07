"""Unit tests for pipeline stage assignment logic.

Tests compute_stage_assignments() for correct layer partitioning,
embedding/lm_head placement, and input validation.
"""

from __future__ import annotations

import pytest

from exo.worker.engines.pytorch.pipeline.stage import (
    StageAssignment,
    compute_stage_assignments,
)


class TestStageAssignmentDataclass:
    """Tests for the StageAssignment frozen dataclass."""

    def test_frozen_immutability(self) -> None:
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=7, has_embedding=True, has_lm_head=False
        )
        with pytest.raises(AttributeError):
            assignment.rank = 1  # type: ignore[misc]

    def test_equality(self) -> None:
        a = StageAssignment(rank=0, start_layer=0, end_layer=7, has_embedding=True, has_lm_head=False)
        b = StageAssignment(rank=0, start_layer=0, end_layer=7, has_embedding=True, has_lm_head=False)
        assert a == b


class TestComputeStageAssignmentsEvenDivision:
    """Tests for even layer division across stages."""

    def test_28_layers_4_nodes(self) -> None:
        """28 layers / 4 nodes = [7, 7, 7, 7]."""
        result = compute_stage_assignments(28, 4)
        assert len(result) == 4
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [7, 7, 7, 7]

    def test_10_layers_2_nodes(self) -> None:
        """10 layers / 2 nodes = [5, 5]."""
        result = compute_stage_assignments(10, 2)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [5, 5]

    def test_12_layers_3_nodes(self) -> None:
        """12 layers / 3 nodes = [4, 4, 4]."""
        result = compute_stage_assignments(12, 3)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [4, 4, 4]


class TestComputeStageAssignmentsUnevenDivision:
    """Tests for uneven layer division — earlier stages get extra."""

    def test_30_layers_4_nodes(self) -> None:
        """30 layers / 4 nodes = [8, 8, 7, 7]."""
        result = compute_stage_assignments(30, 4)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [8, 8, 7, 7]

    def test_28_layers_3_nodes(self) -> None:
        """28 layers / 3 nodes = [10, 9, 9]."""
        result = compute_stage_assignments(28, 3)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [10, 9, 9]

    def test_7_layers_2_nodes(self) -> None:
        """7 layers / 2 nodes = [4, 3]."""
        result = compute_stage_assignments(7, 2)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [4, 3]

    def test_5_layers_4_nodes(self) -> None:
        """5 layers / 4 nodes = [2, 1, 1, 1]."""
        result = compute_stage_assignments(5, 4)
        sizes = [a.end_layer - a.start_layer for a in result]
        assert sizes == [2, 1, 1, 1]


class TestStageContiguity:
    """Tests that stages form a contiguous, non-overlapping partition."""

    def test_layers_are_contiguous(self) -> None:
        result = compute_stage_assignments(30, 4)
        for i in range(len(result) - 1):
            assert result[i].end_layer == result[i + 1].start_layer

    def test_starts_at_zero(self) -> None:
        result = compute_stage_assignments(30, 4)
        assert result[0].start_layer == 0

    def test_ends_at_total_layers(self) -> None:
        result = compute_stage_assignments(30, 4)
        assert result[-1].end_layer == 30

    def test_all_layers_covered(self) -> None:
        result = compute_stage_assignments(30, 4)
        total = sum(a.end_layer - a.start_layer for a in result)
        assert total == 30


class TestEmbeddingAndLmHead:
    """Tests for embedding and LM head placement."""

    def test_rank_0_has_embedding(self) -> None:
        result = compute_stage_assignments(28, 4)
        assert result[0].has_embedding is True
        for assignment in result[1:]:
            assert assignment.has_embedding is False

    def test_last_rank_has_lm_head(self) -> None:
        result = compute_stage_assignments(28, 4)
        assert result[-1].has_lm_head is True
        for assignment in result[:-1]:
            assert assignment.has_lm_head is False

    def test_world_size_2_embedding_and_head(self) -> None:
        result = compute_stage_assignments(10, 2)
        assert result[0].has_embedding is True
        assert result[0].has_lm_head is False
        assert result[1].has_embedding is False
        assert result[1].has_lm_head is True


class TestRankAssignment:
    """Tests that ranks are assigned correctly."""

    def test_ranks_are_sequential(self) -> None:
        result = compute_stage_assignments(28, 4)
        ranks = [a.rank for a in result]
        assert ranks == [0, 1, 2, 3]

    def test_ranks_world_size_3(self) -> None:
        result = compute_stage_assignments(12, 3)
        ranks = [a.rank for a in result]
        assert ranks == [0, 1, 2]


class TestInputValidation:
    """Tests for input validation errors."""

    def test_zero_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="total_layers must be positive"):
            compute_stage_assignments(0, 4)

    def test_negative_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="total_layers must be positive"):
            compute_stage_assignments(-5, 4)

    def test_unsupported_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size must be one of"):
            compute_stage_assignments(28, 5)

    def test_world_size_1_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size must be one of"):
            compute_stage_assignments(28, 1)

    def test_world_size_exceeds_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed total_layers"):
            compute_stage_assignments(2, 4)
