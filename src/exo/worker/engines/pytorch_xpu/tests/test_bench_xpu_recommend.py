"""
Tests for --recommend-layer-distribution CLI flag and format_distribution_comparison.

Verifies:
- The --recommend-layer-distribution flag is parsed correctly into BenchmarkConfig
- format_distribution_comparison produces readable output with expected sections

**Validates: Requirements 6.6**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock torch before loading bench_xpu — allows tests to run without XPU hw
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.__version__ = "2.11.0+xpu"
    _mock_torch.xpu = MagicMock()
    _mock_torch.xpu.is_available = MagicMock(return_value=False)
    _mock_torch.xpu.device_count = MagicMock(return_value=0)
    _mock_torch.long = "torch.int64"
    _mock_torch.bfloat16 = "torch.bfloat16"
    _mock_torch.float16 = "torch.float16"
    _mock_torch.float32 = "torch.float32"
    _mock_torch.Tensor = MagicMock
    sys.modules["torch"] = _mock_torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.backends"] = MagicMock()
    sys.modules["torch.backends.cuda"] = MagicMock()

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_BENCH_XPU_PATH = _ENGINE_DIR / "bench_xpu.py"
_PIPELINE_CONFIG_PATH = _ENGINE_DIR / "pipeline_config.py"
_INSTRUMENTATION_PATH = _ENGINE_DIR / "instrumentation.py"
_STAGE_BALANCING_PATH = _ENGINE_DIR / "stage_balancing.py"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load dependencies first so bench_xpu can import them
_instrumentation_mod = _load_module(
    "instrumentation_recommend_isolated", _INSTRUMENTATION_PATH
)
_config_mod = _load_module(
    "pipeline_config_recommend_isolated", _PIPELINE_CONFIG_PATH
)

# Patch sys.modules so stage_balancing.py and bench_xpu.py can import
sys.modules["exo.worker.engines.pytorch_xpu.instrumentation"] = (
    _instrumentation_mod
)
sys.modules["exo.worker.engines.pytorch_xpu.pipeline_config"] = _config_mod

_stage_balancing_mod = _load_module(
    "stage_balancing_recommend_isolated", _STAGE_BALANCING_PATH
)
sys.modules["exo.worker.engines.pytorch_xpu.stage_balancing"] = (
    _stage_balancing_mod
)

_bench_mod = _load_module("bench_xpu_recommend_isolated", _BENCH_XPU_PATH)

parse_args = _bench_mod.parse_args
BenchmarkConfig = _bench_mod.BenchmarkConfig
format_distribution_comparison = _bench_mod.format_distribution_comparison

PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
PerStageTimingSummary = _stage_balancing_mod.PerStageTimingSummary


# ---------------------------------------------------------------------------
# Tests: --recommend-layer-distribution flag parsing
# ---------------------------------------------------------------------------


class TestRecommendLayerDistributionFlag:
    """Tests for --recommend-layer-distribution CLI argument parsing."""

    def test_flag_defaults_to_false(self) -> None:
        """Without the flag, recommend_layer_distribution is False."""
        config = parse_args([])
        assert config.recommend_layer_distribution is False

    def test_flag_set_to_true(self) -> None:
        """With the flag, recommend_layer_distribution is True."""
        config = parse_args(["--recommend-layer-distribution"])
        assert config.recommend_layer_distribution is True

    def test_flag_combined_with_pipeline_distribution(self) -> None:
        """Flag works alongside --pipeline-layer-distribution."""
        config = parse_args([
            "--pipeline-layer-distribution", "16,16,16,16",
            "--recommend-layer-distribution",
        ])
        assert config.recommend_layer_distribution is True
        assert config.pipeline_layer_distribution is not None
        assert config.pipeline_layer_distribution.layers_per_rank == (
            16, 16, 16, 16
        )

    def test_flag_combined_with_other_args(self) -> None:
        """Flag works alongside other benchmark arguments."""
        config = parse_args([
            "--model_id", "test/model",
            "--warmup", "1",
            "--iterations", "2",
            "--recommend-layer-distribution",
        ])
        assert config.recommend_layer_distribution is True
        assert config.model_id == "test/model"
        assert config.warmup == 1
        assert config.iterations == 2


# ---------------------------------------------------------------------------
# Tests: format_distribution_comparison output
# ---------------------------------------------------------------------------


class TestFormatDistributionComparison:
    """Tests for format_distribution_comparison output formatting."""

    def _make_distribution(
        self, layers_per_rank: tuple[int, ...]
    ) -> PipelineLayerDistribution:
        """Helper to create a PipelineLayerDistribution."""
        return PipelineLayerDistribution(
            layers_per_rank=layers_per_rank,
            total_layer_count=sum(layers_per_rank),
            rank_count=len(layers_per_rank),
        )

    def _make_stage_timing(
        self,
        distribution: PipelineLayerDistribution,
        decode_times: list[float],
    ) -> list[PerStageTimingSummary]:
        """Helper to create per-stage timing summaries."""
        summaries: list[PerStageTimingSummary] = []
        start = 0
        for rank in range(distribution.rank_count):
            layer_count = distribution.layers_per_rank[rank]
            end = start + layer_count
            summaries.append(
                PerStageTimingSummary(
                    rank=rank,
                    start_layer=start,
                    end_layer=end,
                    decode_total_mean_seconds=decode_times[rank],
                    prefill_total_mean_seconds=0.0,
                    layer_count=layer_count,
                )
            )
            start = end
        return summaries

    def test_output_contains_header(self) -> None:
        """Output contains the comparison header."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        current_timing = self._make_stage_timing(
            current, [0.10, 0.12, 0.11, 0.13]
        )
        recommended_timing = self._make_stage_timing(
            recommended, [0.11, 0.11, 0.11, 0.10]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        assert "PIPELINE LAYER DISTRIBUTION COMPARISON" in output

    def test_output_contains_distributions(self) -> None:
        """Output shows both current and recommended distributions."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        current_timing = self._make_stage_timing(
            current, [0.10, 0.12, 0.11, 0.13]
        )
        recommended_timing = self._make_stage_timing(
            recommended, [0.11, 0.11, 0.11, 0.10]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        assert "Current distribution:" in output
        assert "[16,16,16,16]" in output
        assert "Recommended distribution:" in output
        assert "[17,17,16,14]" in output

    def test_output_contains_per_stage_timing(self) -> None:
        """Output contains per-stage timing data for each rank."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        current_timing = self._make_stage_timing(
            current, [0.10, 0.12, 0.11, 0.13]
        )
        recommended_timing = self._make_stage_timing(
            recommended, [0.11, 0.11, 0.11, 0.10]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        # Check that rank numbers appear
        assert "Rank" in output
        # Check timing values appear (at least partially)
        assert "0.1" in output

    def test_output_contains_bottleneck_reduction(self) -> None:
        """Output shows the bottleneck reduction percentage."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        # Current bottleneck is rank 3 at 0.13s
        current_timing = self._make_stage_timing(
            current, [0.10, 0.12, 0.11, 0.13]
        )
        # Recommended bottleneck is 0.11s
        recommended_timing = self._make_stage_timing(
            recommended, [0.11, 0.11, 0.11, 0.10]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        assert "Bottleneck reduction:" in output
        assert "Max stage time (current):" in output
        assert "Max stage time (recommended):" in output
        # Improvement: (0.13 - 0.11) / 0.13 * 100 = 15.4%
        assert "15.4%" in output

    def test_output_with_identical_distributions(self) -> None:
        """Output handles case where current equals recommended."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((16, 16, 16, 16))
        timing = self._make_stage_timing(current, [0.10, 0.10, 0.10, 0.10])

        output = format_distribution_comparison(
            current, recommended, timing, timing
        )

        assert "PIPELINE LAYER DISTRIBUTION COMPARISON" in output
        assert "0.0%" in output

    def test_output_with_zero_timing(self) -> None:
        """Output handles zero timing data gracefully."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        current_timing = self._make_stage_timing(
            current, [0.0, 0.0, 0.0, 0.0]
        )
        recommended_timing = self._make_stage_timing(
            recommended, [0.0, 0.0, 0.0, 0.0]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        assert "N/A" in output

    def test_output_is_multiline_string(self) -> None:
        """Output is a multi-line string with reasonable length."""
        current = self._make_distribution((16, 16, 16, 16))
        recommended = self._make_distribution((17, 17, 16, 14))
        current_timing = self._make_stage_timing(
            current, [0.10, 0.12, 0.11, 0.13]
        )
        recommended_timing = self._make_stage_timing(
            recommended, [0.11, 0.11, 0.11, 0.10]
        )

        output = format_distribution_comparison(
            current, recommended, current_timing, recommended_timing
        )

        lines = output.strip().split("\n")
        # Should have header, distributions, table header, 4 rank rows,
        # separator, max times, improvement, footer
        assert len(lines) >= 10
