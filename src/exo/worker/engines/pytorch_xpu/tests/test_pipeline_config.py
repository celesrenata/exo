"""
Unit tests for pipeline configuration module.

Tests frozen immutability, validation rejects invalid distributions,
and default values are correct.

**Validates: Requirements 6.1, 6.2, 6.6, 6.7, 8.7**
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
    module_name = "pipeline_config_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _PIPELINE_CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_config()
PipelineLayerDistribution = _mod.PipelineLayerDistribution
ChunkedGatedDeltaNetPrefillConfiguration = (
    _mod.ChunkedGatedDeltaNetPrefillConfiguration
)
PytorchXpuOptimizationConfiguration = _mod.PytorchXpuOptimizationConfiguration


# ===========================================================================
# Tests for PipelineLayerDistribution
# ===========================================================================


class TestPipelineLayerDistribution:
    """Test PipelineLayerDistribution immutability and validation."""

    def test_valid_uniform_distribution(self) -> None:
        """A uniform 16-16-16-16 distribution is accepted."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.layers_per_rank == (16, 16, 16, 16)
        assert distribution.total_layer_count == 64
        assert distribution.rank_count == 4

    def test_valid_nonuniform_distribution(self) -> None:
        """A non-uniform distribution that sums correctly is accepted."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(17, 17, 16, 14),
            total_layer_count=64,
            rank_count=4,
        )
        assert distribution.layers_per_rank == (17, 17, 16, 14)

    def test_frozen_immutability(self) -> None:
        """PipelineLayerDistribution instances are immutable."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
            total_layer_count=64,
            rank_count=4,
        )
        with pytest.raises(ValidationError):
            distribution.layers_per_rank = (8, 8, 8, 8)  # type: ignore[misc]

    def test_rejects_sum_not_equal_to_total(self) -> None:
        """Distribution that does not sum to total_layer_count is rejected."""
        with pytest.raises(ValueError, match="sums to"):
            PipelineLayerDistribution(
                layers_per_rank=(16, 16, 16, 15),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_wrong_rank_count(self) -> None:
        """Distribution with wrong number of elements is rejected."""
        with pytest.raises(ValueError, match="elements"):
            PipelineLayerDistribution(
                layers_per_rank=(32, 32),
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

    def test_rejects_empty_distribution(self) -> None:
        """Empty layers_per_rank is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            PipelineLayerDistribution(
                layers_per_rank=(),
                total_layer_count=64,
                rank_count=4,
            )

    def test_rejects_negative_total_layer_count(self) -> None:
        """Negative total_layer_count is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PipelineLayerDistribution(
                layers_per_rank=(16, 16, 16, 16),
                total_layer_count=-1,
                rank_count=4,
            )

    def test_rejects_zero_rank_count(self) -> None:
        """Zero rank_count is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PipelineLayerDistribution(
                layers_per_rank=(16, 16, 16, 16),
                total_layer_count=64,
                rank_count=0,
            )

    def test_two_rank_distribution(self) -> None:
        """A two-rank distribution is valid when configured correctly."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(32, 32),
            total_layer_count=64,
            rank_count=2,
        )
        assert sum(distribution.layers_per_rank) == 64

    def test_defaults_for_total_and_rank(self) -> None:
        """Default total_layer_count is 64 and rank_count is 4."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(16, 16, 16, 16),
        )
        assert distribution.total_layer_count == 64
        assert distribution.rank_count == 4


# ===========================================================================
# Tests for ChunkedGatedDeltaNetPrefillConfiguration
# ===========================================================================


class TestChunkedGatedDeltaNetPrefillConfiguration:
    """Test ChunkedGatedDeltaNetPrefillConfiguration immutability and defaults."""

    def test_default_values(self) -> None:
        """Default configuration has chunked prefill disabled."""
        config = ChunkedGatedDeltaNetPrefillConfiguration()
        assert config.enabled is False
        assert config.chunk_size == 64
        assert config.fallback_on_unsupported_shape is True

    def test_frozen_immutability(self) -> None:
        """ChunkedGatedDeltaNetPrefillConfiguration instances are immutable."""
        config = ChunkedGatedDeltaNetPrefillConfiguration()
        with pytest.raises(ValidationError):
            config.enabled = True  # type: ignore[misc]

    def test_custom_chunk_size(self) -> None:
        """Custom chunk_size is accepted."""
        config = ChunkedGatedDeltaNetPrefillConfiguration(
            enabled=True, chunk_size=128
        )
        assert config.chunk_size == 128

    def test_rejects_zero_chunk_size(self) -> None:
        """Zero chunk_size is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            ChunkedGatedDeltaNetPrefillConfiguration(chunk_size=0)

    def test_rejects_negative_chunk_size(self) -> None:
        """Negative chunk_size is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            ChunkedGatedDeltaNetPrefillConfiguration(chunk_size=-1)


# ===========================================================================
# Tests for PytorchXpuOptimizationConfiguration
# ===========================================================================


class TestPytorchXpuOptimizationConfiguration:
    """Test PytorchXpuOptimizationConfiguration defaults and immutability."""

    def test_default_values_prioritize_correctness(self) -> None:
        """Default configuration prioritizes correctness with opt-in features disabled."""
        config = PytorchXpuOptimizationConfiguration()

        # Enabled by default (safe optimizations)
        assert config.enable_performance_instrumentation is True
        assert config.enable_decode_fast_path is True
        assert config.enable_fast_sampling is True
        assert config.enable_gated_deltanet_persistent_state is True
        assert config.enable_local_shard_loading is True

        # Disabled by default (opt-in until confirmed correct)
        assert config.enable_continuous_batching is False
        assert config.enable_chunked_gated_deltanet_prefill is False

        # Instrumentation flags disabled by default
        assert config.enable_detailed_tracing is False
        assert config.enable_xpu_synchronization_timing is False

        # Numeric defaults
        assert config.maximum_decode_microbatch_size == 8
        assert config.decode_protocol_version == 1

    def test_frozen_immutability(self) -> None:
        """PytorchXpuOptimizationConfiguration instances are immutable."""
        config = PytorchXpuOptimizationConfiguration()
        with pytest.raises(ValidationError):
            config.enable_performance_instrumentation = False  # type: ignore[misc]

    def test_nested_distribution_default(self) -> None:
        """Default pipeline_layer_distribution is uniform 16-16-16-16."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.pipeline_layer_distribution.layers_per_rank == (16, 16, 16, 16)
        assert config.pipeline_layer_distribution.total_layer_count == 64
        assert config.pipeline_layer_distribution.rank_count == 4

    def test_nested_chunked_prefill_default(self) -> None:
        """Default chunked_prefill_configuration has chunked prefill disabled."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.chunked_prefill_configuration.enabled is False
        assert config.chunked_prefill_configuration.chunk_size == 64
        assert config.chunked_prefill_configuration.fallback_on_unsupported_shape is True

    def test_custom_distribution(self) -> None:
        """Custom pipeline_layer_distribution is accepted."""
        custom_distribution = PipelineLayerDistribution(
            layers_per_rank=(17, 17, 16, 14),
            total_layer_count=64,
            rank_count=4,
        )
        config = PytorchXpuOptimizationConfiguration(
            pipeline_layer_distribution=custom_distribution,
        )
        assert config.pipeline_layer_distribution.layers_per_rank == (17, 17, 16, 14)

    def test_enable_detailed_tracing(self) -> None:
        """Detailed tracing can be enabled via configuration."""
        config = PytorchXpuOptimizationConfiguration(
            enable_detailed_tracing=True,
        )
        assert config.enable_detailed_tracing is True

    def test_enable_xpu_synchronization_timing(self) -> None:
        """XPU synchronization timing can be enabled via configuration."""
        config = PytorchXpuOptimizationConfiguration(
            enable_xpu_synchronization_timing=True,
        )
        assert config.enable_xpu_synchronization_timing is True

    def test_rejects_zero_microbatch_size(self) -> None:
        """Zero maximum_decode_microbatch_size is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PytorchXpuOptimizationConfiguration(
                maximum_decode_microbatch_size=0,
            )

    def test_rejects_negative_protocol_version(self) -> None:
        """Negative decode_protocol_version is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            PytorchXpuOptimizationConfiguration(
                decode_protocol_version=-1,
            )

    def test_all_flags_can_be_overridden(self) -> None:
        """All boolean flags can be set to non-default values."""
        config = PytorchXpuOptimizationConfiguration(
            enable_performance_instrumentation=False,
            enable_detailed_tracing=True,
            enable_xpu_synchronization_timing=True,
            enable_decode_fast_path=False,
            enable_fast_sampling=False,
            enable_gated_deltanet_persistent_state=False,
            enable_local_shard_loading=False,
            enable_continuous_batching=True,
            enable_chunked_gated_deltanet_prefill=True,
            maximum_decode_microbatch_size=4,
            decode_protocol_version=2,
        )
        assert config.enable_performance_instrumentation is False
        assert config.enable_detailed_tracing is True
        assert config.enable_xpu_synchronization_timing is True
        assert config.enable_decode_fast_path is False
        assert config.enable_fast_sampling is False
        assert config.enable_gated_deltanet_persistent_state is False
        assert config.enable_local_shard_loading is False
        assert config.enable_continuous_batching is True
        assert config.enable_chunked_gated_deltanet_prefill is True
        assert config.maximum_decode_microbatch_size == 4
        assert config.decode_protocol_version == 2
