"""
Unit tests for pipeline stage distribution helper functions.

Tests parse_layer_distribution_arg, default_layer_distribution, and
the CLI argument parsing for --pipeline-layer-distribution.

**Validates: Requirements 6.1, 6.2, 6.6, 6.7**
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest.mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_CONFIG_PATH = _THIS_DIR.parent / "pipeline_config.py"
_BENCH_XPU_PATH = _THIS_DIR.parent / "bench_xpu.py"


def _load_pipeline_config() -> types.ModuleType:
    """Load pipeline_config.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_config_stage_dist_isolated"
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


def _load_bench_xpu() -> types.ModuleType:
    """Load bench_xpu.py with torch mocked out."""
    module_name = "bench_xpu_stage_dist_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Mock torch so bench_xpu.py can be imported without real torch
    mock_torch = unittest.mock.MagicMock()
    mock_torch.__version__ = "2.11.0+xpu"
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = mock_torch

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, _BENCH_XPU_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    finally:
        # Restore original torch module state
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            del sys.modules["torch"]

    return mod


_config_mod = _load_pipeline_config()
_bench_mod = _load_bench_xpu()

PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
default_layer_distribution = _config_mod.default_layer_distribution
parse_layer_distribution_arg = _config_mod.parse_layer_distribution_arg
parse_args = _bench_mod.parse_args


# ===========================================================================
# Tests for default_layer_distribution
# ===========================================================================


class TestDefaultLayerDistribution:
    """Test default_layer_distribution produces balanced distributions."""

    def test_uniform_64_layers_4_ranks(self) -> None:
        """64 layers across 4 ranks produces (16, 16, 16, 16)."""
        dist = default_layer_distribution(64, 4)
        assert dist.layers_per_rank == (16, 16, 16, 16)
        assert dist.total_layer_count == 64
        assert dist.rank_count == 4

    def test_uniform_32_layers_4_ranks(self) -> None:
        """32 layers across 4 ranks produces (8, 8, 8, 8)."""
        dist = default_layer_distribution(32, 4)
        assert dist.layers_per_rank == (8, 8, 8, 8)

    def test_nonuniform_10_layers_3_ranks(self) -> None:
        """10 layers across 3 ranks: first rank gets extra layer."""
        dist = default_layer_distribution(10, 3)
        assert dist.layers_per_rank == (4, 3, 3)
        assert sum(dist.layers_per_rank) == 10

    def test_nonuniform_7_layers_4_ranks(self) -> None:
        """7 layers across 4 ranks: first 3 ranks get 2, last gets 1."""
        dist = default_layer_distribution(7, 4)
        assert dist.layers_per_rank == (2, 2, 2, 1)
        assert sum(dist.layers_per_rank) == 7

    def test_single_layer_single_rank(self) -> None:
        """1 layer across 1 rank produces (1,)."""
        dist = default_layer_distribution(1, 1)
        assert dist.layers_per_rank == (1,)

    def test_layers_equal_ranks(self) -> None:
        """When layers == ranks, each rank gets exactly 1 layer."""
        dist = default_layer_distribution(4, 4)
        assert dist.layers_per_rank == (1, 1, 1, 1)

    def test_sum_always_equals_total(self) -> None:
        """Sum of layers_per_rank always equals total_layers."""
        for total in [1, 5, 10, 32, 64, 100]:
            for world_size in range(1, min(total + 1, 9)):
                dist = default_layer_distribution(total, world_size)
                assert sum(dist.layers_per_rank) == total

    def test_max_difference_is_one(self) -> None:
        """No two ranks differ by more than 1 layer."""
        dist = default_layer_distribution(65, 4)
        layers = dist.layers_per_rank
        assert max(layers) - min(layers) <= 1

    def test_validates_contiguous(self) -> None:
        """Default distribution always validates as contiguous."""
        dist = default_layer_distribution(64, 4)
        assert dist.validate_contiguous() is True

    def test_rejects_zero_total_layers(self) -> None:
        """Zero total_layers raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            default_layer_distribution(0, 4)

    def test_rejects_zero_world_size(self) -> None:
        """Zero world_size raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            default_layer_distribution(64, 0)

    def test_rejects_negative_total_layers(self) -> None:
        """Negative total_layers raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            default_layer_distribution(-1, 4)

    def test_rejects_negative_world_size(self) -> None:
        """Negative world_size raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            default_layer_distribution(64, -1)


# ===========================================================================
# Tests for parse_layer_distribution_arg
# ===========================================================================


class TestParseLayerDistributionArg:
    """Test parse_layer_distribution_arg with valid and invalid inputs."""

    def test_valid_uniform(self) -> None:
        """Parses '16,16,16,16' correctly."""
        dist = parse_layer_distribution_arg("16,16,16,16", 64, 4)
        assert dist.layers_per_rank == (16, 16, 16, 16)
        assert dist.total_layer_count == 64
        assert dist.rank_count == 4

    def test_valid_nonuniform(self) -> None:
        """Parses '17,17,16,14' correctly."""
        dist = parse_layer_distribution_arg("17,17,16,14", 64, 4)
        assert dist.layers_per_rank == (17, 17, 16, 14)

    def test_valid_with_spaces(self) -> None:
        """Parses '16, 16, 16, 16' with spaces correctly."""
        dist = parse_layer_distribution_arg("16, 16, 16, 16", 64, 4)
        assert dist.layers_per_rank == (16, 16, 16, 16)

    def test_valid_with_leading_trailing_spaces(self) -> None:
        """Parses ' 16,16,16,16 ' with leading/trailing spaces."""
        dist = parse_layer_distribution_arg(" 16,16,16,16 ", 64, 4)
        assert dist.layers_per_rank == (16, 16, 16, 16)

    def test_valid_two_ranks(self) -> None:
        """Parses '32,32' for two ranks."""
        dist = parse_layer_distribution_arg("32,32", 64, 2)
        assert dist.layers_per_rank == (32, 32)

    def test_rejects_wrong_rank_count(self) -> None:
        """Rejects distribution with wrong number of elements."""
        with pytest.raises(ValueError, match="elements"):
            parse_layer_distribution_arg("16,16,16", 64, 4)

    def test_rejects_wrong_sum(self) -> None:
        """Rejects distribution that does not sum to total_layers."""
        with pytest.raises(ValueError, match="sums to"):
            parse_layer_distribution_arg("16,16,16,15", 64, 4)

    def test_rejects_non_integer(self) -> None:
        """Rejects non-integer values."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_layer_distribution_arg("16,16,abc,16", 64, 4)

    def test_rejects_float_values(self) -> None:
        """Rejects float values."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_layer_distribution_arg("16.5,16,16,15.5", 64, 4)

    def test_rejects_zero_layer_stage(self) -> None:
        """Rejects distribution with a zero-layer stage."""
        with pytest.raises(ValueError, match="at least one layer"):
            parse_layer_distribution_arg("20,20,24,0", 64, 4)

    def test_rejects_negative_layer_count(self) -> None:
        """Rejects distribution with a negative layer count."""
        with pytest.raises(ValueError, match="at least one layer"):
            parse_layer_distribution_arg("20,20,25,-1", 64, 4)

    def test_rejects_empty_string(self) -> None:
        """Rejects empty string."""
        with pytest.raises(ValueError):
            parse_layer_distribution_arg("", 64, 4)

    def test_validates_contiguous(self) -> None:
        """Parsed distribution validates as contiguous."""
        dist = parse_layer_distribution_arg("17,17,16,14", 64, 4)
        assert dist.validate_contiguous() is True


# ===========================================================================
# Tests for CLI argument parsing
# ===========================================================================


class TestCliLayerDistributionArg:
    """Test --pipeline-layer-distribution CLI argument parsing."""

    def test_no_distribution_arg_returns_none(self) -> None:
        """Without --pipeline-layer-distribution, field is None."""
        config = parse_args(["--model_id", "test-model"])
        assert config.pipeline_layer_distribution is None

    def test_valid_distribution_arg(self) -> None:
        """Valid --pipeline-layer-distribution is parsed into a PipelineLayerDistribution."""
        config = parse_args(["--pipeline-layer-distribution", "16,16,16,16"])
        assert config.pipeline_layer_distribution is not None
        assert config.pipeline_layer_distribution.layers_per_rank == (
            16,
            16,
            16,
            16,
        )
        assert config.pipeline_layer_distribution.total_layer_count == 64
        assert config.pipeline_layer_distribution.rank_count == 4

    def test_nonuniform_distribution_arg(self) -> None:
        """Non-uniform distribution is parsed correctly."""
        config = parse_args(
            ["--pipeline-layer-distribution", "17,17,16,14"]
        )
        assert config.pipeline_layer_distribution is not None
        assert config.pipeline_layer_distribution.layers_per_rank == (
            17,
            17,
            16,
            14,
        )

    def test_two_rank_distribution_arg(self) -> None:
        """Two-rank distribution is parsed correctly."""
        config = parse_args(["--pipeline-layer-distribution", "32,32"])
        assert config.pipeline_layer_distribution is not None
        assert config.pipeline_layer_distribution.layers_per_rank == (32, 32)
        assert config.pipeline_layer_distribution.rank_count == 2
        assert config.pipeline_layer_distribution.total_layer_count == 64

    def test_invalid_distribution_exits(self) -> None:
        """Invalid --pipeline-layer-distribution causes sys.exit."""
        with pytest.raises(SystemExit):
            parse_args(["--pipeline-layer-distribution", "abc,def"])

    def test_zero_layer_stage_exits(self) -> None:
        """Distribution with zero-layer stage causes sys.exit."""
        with pytest.raises(SystemExit):
            parse_args(["--pipeline-layer-distribution", "20,20,24,0"])
