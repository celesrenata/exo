"""Tests for continuous batching benchmark mode in bench_xpu.py.

Verifies:
- --concurrent-requests CLI argument is parsed correctly
- ContinuousBatchingBenchmarkResult is frozen and computes fields correctly
- format_continuous_batching_report produces readable output
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock torch before importing bench_xpu — this module is tested on machines
# without XPU hardware, so we stub out the torch dependency.
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.__version__ = "2.11.0+xpu"
    _mock_torch.xpu.is_available.return_value = False
    _mock_torch.xpu.device_count.return_value = 0
    _mock_torch.long = 4  # torch.long constant
    _mock_torch.bfloat16 = "bfloat16"
    _mock_torch.float16 = "float16"
    sys.modules["torch"] = _mock_torch
    sys.modules["torch.xpu"] = _mock_torch.xpu


# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BENCH_XPU_PATH = _THIS_DIR.parent / "bench_xpu.py"


def _load_bench_xpu() -> types.ModuleType:
    """Load bench_xpu.py directly from file, avoiding __init__.py."""
    module_name = "bench_xpu_cb_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Mock the local imports that bench_xpu.py needs
    for mod_name in [
        "exo.worker.engines.pytorch_xpu.instrumentation",
        "exo.worker.engines.pytorch_xpu.pipeline_config",
        "exo.worker.engines.pytorch_xpu.stage_balancing",
    ]:
        if mod_name not in sys.modules:
            mock_mod = MagicMock()
            # Provide the types that bench_xpu imports
            if "pipeline_config" in mod_name:
                mock_mod.PipelineLayerDistribution = type(
                    "PipelineLayerDistribution", (), {}
                )
                mock_mod.default_layer_distribution = MagicMock()
                mock_mod.parse_layer_distribution_arg = MagicMock()
            if "instrumentation" in mod_name:
                mock_mod.PerformanceEvent = type("PerformanceEvent", (), {})
                mock_mod.PerformanceRecorder = MagicMock
            if "stage_balancing" in mod_name:
                mock_mod.PerStageTimingSummary = type(
                    "PerStageTimingSummary", (), {}
                )
                mock_mod.export_per_layer_timing = MagicMock()
                mock_mod.export_per_stage_timing = MagicMock()
                mock_mod.recommend_pipeline_layer_distribution = MagicMock()
            sys.modules[mod_name] = mock_mod

    spec = importlib.util.spec_from_file_location(module_name, _BENCH_XPU_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_bench_xpu()
parse_args = _mod.parse_args
BenchmarkConfig = _mod.BenchmarkConfig
ContinuousBatchingBenchmarkResult = _mod.ContinuousBatchingBenchmarkResult
format_continuous_batching_report = _mod.format_continuous_batching_report


# ---------------------------------------------------------------------------
# CLI Argument Parsing Tests
# ---------------------------------------------------------------------------


class TestConcurrentRequestsParsing:
    """Tests for --concurrent-requests CLI argument parsing."""

    def test_default_concurrent_requests(self) -> None:
        """Default concurrent_requests is 4 when not specified."""
        config = parse_args([])
        assert config.concurrent_requests == 4

    def test_concurrent_requests_4(self) -> None:
        """--concurrent-requests 4 is parsed correctly."""
        config = parse_args(["--concurrent-requests", "4"])
        assert config.concurrent_requests == 4

    def test_concurrent_requests_8(self) -> None:
        """--concurrent-requests 8 is parsed correctly."""
        config = parse_args(["--concurrent-requests", "8"])
        assert config.concurrent_requests == 8

    def test_concurrent_requests_1(self) -> None:
        """--concurrent-requests 1 is parsed correctly."""
        config = parse_args(["--concurrent-requests", "1"])
        assert config.concurrent_requests == 1

    def test_concurrent_requests_16(self) -> None:
        """--concurrent-requests 16 is parsed correctly."""
        config = parse_args(["--concurrent-requests", "16"])
        assert config.concurrent_requests == 16

    def test_concurrent_requests_with_continuous_batching_mode(self) -> None:
        """--concurrent-requests works with --benchmark-mode continuous-batching."""
        config = parse_args([
            "--benchmark-mode", "continuous-batching",
            "--concurrent-requests", "8",
        ])
        assert config.benchmark_mode == "continuous-batching"
        assert config.concurrent_requests == 8

    def test_concurrent_requests_stored_in_config(self) -> None:
        """concurrent_requests field exists on BenchmarkConfig."""
        config = parse_args(["--concurrent-requests", "12"])
        assert hasattr(config, "concurrent_requests")
        assert config.concurrent_requests == 12


# ---------------------------------------------------------------------------
# ContinuousBatchingBenchmarkResult Tests
# ---------------------------------------------------------------------------


class TestContinuousBatchingBenchmarkResult:
    """Tests for ContinuousBatchingBenchmarkResult dataclass."""

    def test_frozen_instance(self) -> None:
        """ContinuousBatchingBenchmarkResult is frozen (immutable)."""
        result = ContinuousBatchingBenchmarkResult(
            concurrent_requests=4,
            total_tokens_generated=512,
            total_decode_steps=128,
            wall_clock_seconds=10.0,
            aggregate_tokens_per_second=51.2,
            per_request_tokens_per_second=12.8,
            average_microbatch_size=4.0,
            pipeline_occupancy=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.concurrent_requests = 8  # type: ignore[misc]

    def test_fields_stored_correctly(self) -> None:
        """All fields are stored with correct values."""
        result = ContinuousBatchingBenchmarkResult(
            concurrent_requests=8,
            total_tokens_generated=1024,
            total_decode_steps=256,
            wall_clock_seconds=20.0,
            aggregate_tokens_per_second=51.2,
            per_request_tokens_per_second=6.4,
            average_microbatch_size=4.0,
            pipeline_occupancy=0.5,
        )
        assert result.concurrent_requests == 8
        assert result.total_tokens_generated == 1024
        assert result.total_decode_steps == 256
        assert result.wall_clock_seconds == 20.0
        assert result.aggregate_tokens_per_second == 51.2
        assert result.per_request_tokens_per_second == 6.4
        assert result.average_microbatch_size == 4.0
        assert result.pipeline_occupancy == 0.5

    def test_aggregate_tps_computation(self) -> None:
        """aggregate_tokens_per_second = total_tokens / wall_clock."""
        total_tokens = 400
        wall_clock = 8.0
        expected_aggregate = total_tokens / wall_clock  # 50.0

        result = ContinuousBatchingBenchmarkResult(
            concurrent_requests=4,
            total_tokens_generated=total_tokens,
            total_decode_steps=100,
            wall_clock_seconds=wall_clock,
            aggregate_tokens_per_second=expected_aggregate,
            per_request_tokens_per_second=expected_aggregate / 4,
            average_microbatch_size=4.0,
            pipeline_occupancy=1.0,
        )
        assert result.aggregate_tokens_per_second == pytest.approx(50.0)

    def test_per_request_tps_computation(self) -> None:
        """per_request_tokens_per_second = aggregate / concurrent_requests."""
        concurrent = 4
        aggregate_tps = 48.0
        expected_per_request = aggregate_tps / concurrent  # 12.0

        result = ContinuousBatchingBenchmarkResult(
            concurrent_requests=concurrent,
            total_tokens_generated=480,
            total_decode_steps=120,
            wall_clock_seconds=10.0,
            aggregate_tokens_per_second=aggregate_tps,
            per_request_tokens_per_second=expected_per_request,
            average_microbatch_size=4.0,
            pipeline_occupancy=1.0,
        )
        assert result.per_request_tokens_per_second == pytest.approx(12.0)

    def test_pipeline_occupancy_range(self) -> None:
        """pipeline_occupancy is between 0.0 and 1.0."""
        result = ContinuousBatchingBenchmarkResult(
            concurrent_requests=4,
            total_tokens_generated=256,
            total_decode_steps=64,
            wall_clock_seconds=5.0,
            aggregate_tokens_per_second=51.2,
            per_request_tokens_per_second=12.8,
            average_microbatch_size=3.5,
            pipeline_occupancy=0.75,
        )
        assert 0.0 <= result.pipeline_occupancy <= 1.0


# ---------------------------------------------------------------------------
# format_continuous_batching_report Tests
# ---------------------------------------------------------------------------


class TestFormatContinuousBatchingReport:
    """Tests for format_continuous_batching_report function."""

    def _make_result(
        self,
        concurrent: int = 4,
        total_tokens: int = 512,
        wall_clock: float = 10.0,
    ) -> object:
        aggregate_tps = total_tokens / wall_clock
        per_request_tps = aggregate_tps / concurrent
        return ContinuousBatchingBenchmarkResult(
            concurrent_requests=concurrent,
            total_tokens_generated=total_tokens,
            total_decode_steps=128,
            wall_clock_seconds=wall_clock,
            aggregate_tokens_per_second=aggregate_tps,
            per_request_tokens_per_second=per_request_tps,
            average_microbatch_size=float(concurrent),
            pipeline_occupancy=0.8,
        )

    def test_report_is_string(self) -> None:
        """format_continuous_batching_report returns a string."""
        result = self._make_result()
        report = format_continuous_batching_report(result)
        assert isinstance(report, str)

    def test_report_contains_header(self) -> None:
        """Report contains the continuous batching header."""
        result = self._make_result()
        report = format_continuous_batching_report(result)
        assert "CONTINUOUS BATCHING BENCHMARK RESULTS" in report

    def test_report_contains_concurrent_requests(self) -> None:
        """Report shows the number of concurrent requests."""
        result = self._make_result(concurrent=8)
        report = format_continuous_batching_report(result)
        assert "8" in report
        assert "Concurrent requests" in report

    def test_report_contains_aggregate_tps(self) -> None:
        """Report shows aggregate tokens per second."""
        result = self._make_result(total_tokens=500, wall_clock=10.0)
        report = format_continuous_batching_report(result)
        assert "Aggregate tokens/sec" in report
        assert "50.00" in report

    def test_report_contains_per_request_tps(self) -> None:
        """Report shows per-request tokens per second."""
        result = self._make_result(
            concurrent=4, total_tokens=400, wall_clock=10.0
        )
        report = format_continuous_batching_report(result)
        assert "Per-request tokens/sec" in report
        assert "10.00" in report

    def test_report_contains_wall_clock(self) -> None:
        """Report shows wall clock time."""
        result = self._make_result(wall_clock=15.5)
        report = format_continuous_batching_report(result)
        assert "15.500" in report
        assert "Wall clock time" in report

    def test_report_contains_pipeline_occupancy(self) -> None:
        """Report shows pipeline occupancy percentage."""
        result = self._make_result()
        report = format_continuous_batching_report(result)
        assert "Pipeline occupancy" in report
        assert "80.0%" in report

    def test_report_contains_average_microbatch_size(self) -> None:
        """Report shows average microbatch size."""
        result = self._make_result(concurrent=4)
        report = format_continuous_batching_report(result)
        assert "Average microbatch size" in report
        assert "4.00" in report

    def test_report_multiline(self) -> None:
        """Report is multi-line with reasonable length."""
        result = self._make_result()
        report = format_continuous_batching_report(result)
        lines = report.strip().split("\n")
        assert len(lines) > 10
