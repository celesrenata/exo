"""
Unit tests for PipelineStageAssignment and PipelineLayerDistribution.

Tests valid distributions, invalid distributions, get_stage_assignment(),
validate_contiguous(), and embedding/lm_head ownership.

**Validates: Requirements 1.4, 1.5, 1.6, 6.1, 6.2, 6.8**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_CONFIG_PATH = _THIS_DIR.parent / "pipeline_config.py"


def _load_pipeline_config() -> types.ModuleType:
    """Load pipeline_config.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_config_layer_dist_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, _PIPELINE_CONFIG_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_config()
PipelineStageAssignment = _mod.PipelineStageAssignment
PipelineLayerDistribution = _mod.PipelineLayerDistribution


# ===========================================================================
# Tests for PipelineStageAssignment
# ===========================================================================


class TestPipelineStageAssignment:
    """Test PipelineStageAssignment validation and properties."""

    def test_valid_assignment(self) -> None:
        """A valid stage assignment is accepted."""
        assignment = PipelineStageAssignment(
            rank=0,
            start_layer=0,
            end_layer=16,
            owns_embedding=True,
            owns_lm_head=False,
        )
        assert assignment.rank == 0
        assert assignment.start_layer == 0
        assert assignment.end_layer == 16
        assert assignment.owns_embedding is True
        assert assignment.owns_lm_head is False

    def test_num_local_layers(self) -> None:
        """num_local_layers returns end_layer - start_layer."""
        assignment = PipelineStageAssignment(
            rank=1,
            start_layer=16,
            end_layer=32,
            owns_embedding=False,
            owns_lm_head=False,
        )
        assert assignment.num_local_layers == 16

    def test_frozen_immutability(self) -> None:
        """PipelineStageAssignment instances are immutable."""
        assignment = PipelineStageAssignment(
            rank=0,
            start_layer=0,
            end_layer=16,
            owns_embedding=True,
            owns_lm_head=False,
        )
        with pytest.raises(ValidationError):
            assignment.rank = 1  # type: ignore[misc]

    def test_rejects_negative_rank(self) -> None:
        """Negative rank is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            PipelineStageAssignment(
                rank=-1,
                start_layer=0,
                end_layer=16,
                owns_embedding=True,
                owns_lm_head=False,
            )

    def test_rejects_start_layer_equal_to_end_layer(self) -> None:
        """start_layer == end_layer is rejected (zero-width range)."""
        with pytest.raises(ValueError, match="must be less than"):
            PipelineStageAssignment(
                rank=0,
                start_layer=16,
                end_layer=16,
                owns_embedding=True,
                owns_lm_head=False,
            )

    def test_rejects_start_layer_greater_than_end_layer(self) -> None:
        """start_layer > end_layer is rejected."""
        with pytest.raises(ValueError, match="must be less than"):
            PipelineStageAssignment(
                rank=0,
                start_layer=20,
                end_layer=16,
                owns_embedding=True,
                owns_lm_head=False,
            )


# ===========================================================================
# Tests for PipelineLayerDistribution — valid distributions
# ===========================================================================


class TestPipelineLayerDistributionValid:
    """Test valid PipelineLayerDistribution configurations."""

    def test_balanced_distribution(self) -> None:
        """Balanced [16,16,16,16] distribution is accepted."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.layers_per_rank == (16, 16, 16, 16)
        assert distribution.total_layer_count == 64
        assert distribution.rank_count == 4

    def test_unbalanced_distribution(self) -> None:
        """Unbalanced [17,17,16,14] distribution is accepted."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(17, 17, 16, 14),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.layers_per_rank == (17, 17, 16, 14)
        assert sum(distribution.layers_per_rank) == 64

    def test_single_rank_distribution(self) -> None:
        """Single rank owning all layers is valid."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(64,),
            total_layer_count=64,
            rank_count=1,
        )
        assert distribution.rank_count == 1
        assert distribution.layers_per_rank == (64,)

    def test_validate_contiguous_balanced(self) -> None:
        """Balanced distribution is contiguous."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.validate_contiguous() is True

    def test_validate_contiguous_unbalanced(self) -> None:
        """Unbalanced distribution is contiguous."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(17, 17, 16, 14),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.validate_contiguous() is True


# ===========================================================================
# Tests for PipelineLayerDistribution — invalid distributions
# ===========================================================================


class TestPipelineLayerDistributionInvalid:
    """Test invalid PipelineLayerDistribution configurations are rejected."""

    def test_rejects_sum_not_equal_to_total(self) -> None:
        """Distribution that does not sum to total_layer_count is rejected."""
        with pytest.raises(ValueError, match="sums to"):
            PipelineLayerDistribution(
                layers_per_rank=(16, 16, 16, 15),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_zero_layer_stage(self) -> None:
        """Distribution with a zero-layer stage is rejected."""
        with pytest.raises(ValueError, match="at least one layer"):
            PipelineLayerDistribution(
                layers_per_rank=(20, 20, 24, 0),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_negative_layer_count(self) -> None:
        """Distribution with a negative layer count is rejected."""
        with pytest.raises(ValueError, match="at least one layer"):
            PipelineLayerDistribution(
                layers_per_rank=(20, 20, 25, -1),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_rank_count_mismatch(self) -> None:
        """Distribution with wrong number of elements is rejected."""
        with pytest.raises(ValueError, match="elements"):
            PipelineLayerDistribution(
                layers_per_rank=(32, 32),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_zero_total_layer_count(self) -> None:
        """Zero total_layer_count is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PipelineLayerDistribution(
                layers_per_rank=(1,),
                total_layer_count=0,
                rank_count=1,
            )

    def test_rejects_zero_rank_count(self) -> None:
        """Zero rank_count is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PipelineLayerDistribution(
                layers_per_rank=(16, 16, 16, 16),
                total_layer_count=64,
                rank_count=0,
            )


# ===========================================================================
# Tests for get_stage_assignment()
# ===========================================================================


class TestGetStageAssignment:
    """Test PipelineLayerDistribution.get_stage_assignment() method."""

    def test_balanced_rank_0(self) -> None:
        """Rank 0 in balanced distribution gets layers [0, 16)."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        assert assignment.rank == 0
        assert assignment.start_layer == 0
        assert assignment.end_layer == 16
        assert assignment.num_local_layers == 16

    def test_balanced_rank_1(self) -> None:
        """Rank 1 in balanced distribution gets layers [16, 32)."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(1)
        assert assignment.rank == 1
        assert assignment.start_layer == 16
        assert assignment.end_layer == 32

    def test_balanced_rank_3(self) -> None:
        """Rank 3 in balanced distribution gets layers [48, 64)."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(3)
        assert assignment.rank == 3
        assert assignment.start_layer == 48
        assert assignment.end_layer == 64

    def test_unbalanced_ranges(self) -> None:
        """Unbalanced [17,17,16,14] produces correct contiguous ranges."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(17, 17, 16, 14),
            total_layer_count=64,
            rank_count=4,
        )
        a0 = distribution.get_stage_assignment(0)
        a1 = distribution.get_stage_assignment(1)
        a2 = distribution.get_stage_assignment(2)
        a3 = distribution.get_stage_assignment(3)

        assert a0.start_layer == 0 and a0.end_layer == 17
        assert a1.start_layer == 17 and a1.end_layer == 34
        assert a2.start_layer == 34 and a2.end_layer == 50
        assert a3.start_layer == 50 and a3.end_layer == 64

    def test_rejects_rank_out_of_range(self) -> None:
        """Rank >= rank_count is rejected."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        with pytest.raises(ValueError, match="must be in"):
            distribution.get_stage_assignment(4)

    def test_rejects_negative_rank(self) -> None:
        """Negative rank is rejected."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        with pytest.raises(ValueError, match="must be in"):
            distribution.get_stage_assignment(-1)


# ===========================================================================
# Tests for embedding/lm_head ownership
# ===========================================================================


class TestOwnership:
    """Test embedding and lm_head ownership assignment."""

    def test_rank_0_owns_embedding(self) -> None:
        """Rank 0 owns the token embedding."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        assert assignment.owns_embedding is True
        assert assignment.owns_lm_head is False

    def test_last_rank_owns_lm_head(self) -> None:
        """Last rank owns the final normalization and lm_head."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(3)
        assert assignment.owns_embedding is False
        assert assignment.owns_lm_head is True

    def test_middle_ranks_own_neither(self) -> None:
        """Middle ranks own neither embedding nor lm_head."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        for rank in (1, 2):
            assignment = distribution.get_stage_assignment(rank)
            assert assignment.owns_embedding is False
            assert assignment.owns_lm_head is False

    def test_single_rank_owns_both(self) -> None:
        """Single rank owns both embedding and lm_head."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(64,),
            total_layer_count=64,
            rank_count=1,
        )
        assignment = distribution.get_stage_assignment(0)
        assert assignment.owns_embedding is True
        assert assignment.owns_lm_head is True

    def test_two_rank_ownership(self) -> None:
        """In a two-rank setup, rank 0 owns embedding, rank 1 owns lm_head."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(32, 32),
            total_layer_count=64,
            rank_count=2,
        )
        a0 = distribution.get_stage_assignment(0)
        a1 = distribution.get_stage_assignment(1)
        assert a0.owns_embedding is True
        assert a0.owns_lm_head is False
        assert a1.owns_embedding is False
        assert a1.owns_lm_head is True
