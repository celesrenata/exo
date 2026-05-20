"""
Unit tests for the decode benchmark harness.

Tests verify:
- DecodeMetrics immutability and field correctness
- DecodeBenchmark token recording and metric computation
- GPU idle percentage calculation from frequency samples
- Coefficient of variation computation
- Integration with PerformanceRecorder event extraction
- Performance target validation

**Validates: Requirements 10.1, 10.2, 10.4**
"""

from __future__ import annotations

import time

import pytest

from exo.worker.engines.pytorch_xpu.benchmark_decode import (
    DecodeBenchmark,
    DecodeMetrics,
    PerformanceTargets,
    compute_gpu_idle_percentage,
    compute_latency_coefficient_of_variation,
    validate_performance_targets,
)
from exo.worker.engines.pytorch_xpu.instrumentation import (
    PerformanceRecorder,
)


class TestDecodeMetricsImmutability:
    """Test that DecodeMetrics is immutable (frozen=True)."""

    def test_cannot_modify_tokens_per_second(self) -> None:
        """Verify tokens_per_second cannot be modified after creation."""
        from pydantic import ValidationError

        metrics = DecodeMetrics(
            tokens_per_second=10.0,
            gpu_idle_percentage=5.0,
            latency_cv=0.15,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=100.0,
            min_token_latency_ms=80.0,
            max_token_latency_ms=120.0,
            measurement_duration_seconds=4.5,
        )
        with pytest.raises(ValidationError):
            metrics.tokens_per_second = 20.0

    def test_all_fields_populated(self) -> None:
        """Verify all fields are accessible after creation."""
        metrics = DecodeMetrics(
            tokens_per_second=12.5,
            gpu_idle_percentage=8.3,
            latency_cv=0.12,
            total_tokens_generated=55,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=80.0,
            min_token_latency_ms=60.0,
            max_token_latency_ms=110.0,
            measurement_duration_seconds=4.0,
        )
        assert metrics.tokens_per_second == 12.5
        assert metrics.gpu_idle_percentage == 8.3
        assert metrics.latency_cv == 0.12
        assert metrics.total_tokens_generated == 55
        assert metrics.warmup_tokens_excluded == 5
        assert metrics.mean_token_latency_ms == 80.0
        assert metrics.min_token_latency_ms == 60.0
        assert metrics.max_token_latency_ms == 110.0
        assert metrics.measurement_duration_seconds == 4.0


class TestComputeGpuIdlePercentage:
    """Test GPU idle percentage computation from frequency samples."""

    def test_all_idle(self) -> None:
        """All samples below threshold yields 100% idle."""
        samples = [100, 200, 250, 300]
        result = compute_gpu_idle_percentage(samples, idle_threshold_mhz=300)
        assert result == 100.0

    def test_none_idle(self) -> None:
        """All samples above threshold yields 0% idle."""
        samples = [1200, 1400, 1600, 1800]
        result = compute_gpu_idle_percentage(samples, idle_threshold_mhz=300)
        assert result == 0.0

    def test_mixed_samples(self) -> None:
        """Mix of idle and active samples yields correct percentage."""
        # 2 idle (200, 300), 2 active (800, 1200)
        samples = [200, 800, 300, 1200]
        result = compute_gpu_idle_percentage(samples, idle_threshold_mhz=300)
        assert result == 50.0

    def test_empty_samples(self) -> None:
        """Empty sample list yields 0% idle."""
        result = compute_gpu_idle_percentage([])
        assert result == 0.0

    def test_single_idle_sample(self) -> None:
        """Single idle sample yields 100%."""
        result = compute_gpu_idle_percentage([100], idle_threshold_mhz=300)
        assert result == 100.0

    def test_single_active_sample(self) -> None:
        """Single active sample yields 0%."""
        result = compute_gpu_idle_percentage([1500], idle_threshold_mhz=300)
        assert result == 0.0


class TestComputeLatencyCoefficientOfVariation:
    """Test coefficient of variation computation."""

    def test_identical_latencies_yield_zero_cv(self) -> None:
        """Identical latencies have zero variation."""
        latencies = [100.0, 100.0, 100.0, 100.0, 100.0]
        cv = compute_latency_coefficient_of_variation(latencies)
        assert cv == 0.0

    def test_known_cv_value(self) -> None:
        """Verify CV computation against a known result."""
        # Mean = 100, std = 10, CV = 0.1
        # Using values that produce exactly std=10: [90, 110] repeated
        latencies = [90.0, 110.0, 90.0, 110.0]
        cv = compute_latency_coefficient_of_variation(latencies)
        # With sample std (n-1): variance = (4 * 100) / 3 = 133.33, std ≈ 11.55
        # CV ≈ 11.55 / 100 ≈ 0.1155
        assert 0.10 < cv < 0.13

    def test_high_variation(self) -> None:
        """High variation produces CV > 0.20."""
        latencies = [50.0, 200.0, 50.0, 200.0, 50.0]
        cv = compute_latency_coefficient_of_variation(latencies)
        assert cv > 0.20

    def test_single_sample_returns_zero(self) -> None:
        """Single sample cannot compute CV, returns 0."""
        cv = compute_latency_coefficient_of_variation([100.0])
        assert cv == 0.0

    def test_empty_list_returns_zero(self) -> None:
        """Empty list returns 0."""
        cv = compute_latency_coefficient_of_variation([])
        assert cv == 0.0

    def test_two_samples(self) -> None:
        """Two samples produce a valid CV."""
        latencies = [80.0, 120.0]
        cv = compute_latency_coefficient_of_variation(latencies)
        # Mean = 100, sample std = sqrt((400+400)/1) = sqrt(800) ≈ 28.28
        # CV ≈ 28.28 / 100 ≈ 0.2828
        assert 0.25 < cv < 0.30


class TestDecodeBenchmark:
    """Test the DecodeBenchmark collector class."""

    def test_initial_state(self) -> None:
        """Verify initial state after construction."""
        benchmark = DecodeBenchmark(warmup_tokens=3)
        assert benchmark.warmup_tokens == 3
        assert benchmark.token_count == 0
        assert benchmark.gpu_frequency_sample_count == 0

    def test_record_tokens(self) -> None:
        """Verify token recording increments count."""
        benchmark = DecodeBenchmark()
        benchmark.start()
        benchmark.record_token()
        benchmark.record_token()
        benchmark.record_token()
        assert benchmark.token_count == 3

    def test_record_gpu_frequency(self) -> None:
        """Verify GPU frequency recording increments count."""
        benchmark = DecodeBenchmark()
        benchmark.start()
        benchmark.record_gpu_frequency(1200)
        benchmark.record_gpu_frequency(1400)
        assert benchmark.gpu_frequency_sample_count == 2

    def test_compute_metrics_no_tokens_raises(self) -> None:
        """Verify compute_metrics raises when no tokens recorded."""
        benchmark = DecodeBenchmark()
        benchmark.start()
        with pytest.raises(ValueError, match="No tokens recorded"):
            benchmark.compute_metrics()

    def test_compute_metrics_basic(self) -> None:
        """Verify basic metric computation with simulated tokens."""
        benchmark = DecodeBenchmark(warmup_tokens=2)
        benchmark.start()

        # Simulate 5 tokens at ~10ms intervals
        for _ in range(5):
            time.sleep(0.01)  # ~10ms per token
            benchmark.record_token()

        metrics = benchmark.compute_metrics()
        assert metrics.total_tokens_generated == 5
        assert metrics.warmup_tokens_excluded == 2
        # After warmup, 3 tokens remain
        # At ~10ms per token, expect ~100 tok/s (but sleep is imprecise)
        assert metrics.tokens_per_second > 0.0
        assert metrics.mean_token_latency_ms > 0.0
        assert metrics.min_token_latency_ms > 0.0
        assert metrics.max_token_latency_ms >= metrics.min_token_latency_ms
        assert metrics.measurement_duration_seconds > 0.0

    def test_compute_metrics_with_gpu_samples(self) -> None:
        """Verify GPU idle percentage is included in metrics."""
        benchmark = DecodeBenchmark(warmup_tokens=0, idle_threshold_mhz=300)
        benchmark.start()

        # Record tokens
        time.sleep(0.005)
        benchmark.record_token()
        time.sleep(0.005)
        benchmark.record_token()

        # Record GPU frequencies: 1 idle, 1 active
        benchmark.record_gpu_frequency(200)
        benchmark.record_gpu_frequency(1500)

        metrics = benchmark.compute_metrics()
        assert metrics.gpu_idle_percentage == 50.0

    def test_get_per_token_latencies(self) -> None:
        """Verify per-token latency extraction."""
        benchmark = DecodeBenchmark()
        benchmark.start()

        time.sleep(0.01)
        benchmark.record_token()
        time.sleep(0.02)
        benchmark.record_token()

        latencies = benchmark.get_per_token_latencies_ms()
        assert len(latencies) == 2
        # First token ~10ms, second ~20ms (approximate due to sleep imprecision)
        assert latencies[0] > 5.0  # at least 5ms
        assert latencies[1] > 10.0  # at least 10ms

    def test_warmup_exceeds_tokens(self) -> None:
        """Verify graceful handling when warmup exceeds token count."""
        benchmark = DecodeBenchmark(warmup_tokens=100)
        benchmark.start()

        time.sleep(0.005)
        benchmark.record_token()
        time.sleep(0.005)
        benchmark.record_token()

        # Should not raise — falls back to using all latencies
        metrics = benchmark.compute_metrics()
        assert metrics.total_tokens_generated == 2
        # When warmup exceeds tokens, effective_warmup = total - 1 = 1
        assert metrics.warmup_tokens_excluded == 1


class TestFromPerformanceRecorder:
    """Test extraction of metrics from PerformanceRecorder."""

    def test_extracts_decode_token_latencies(self) -> None:
        """Verify metrics extraction from recorder's decode events."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)

        # Simulate decode token latency spans
        for _ in range(10):
            with recorder.span("end_to_end_token_latency", mode="decode"):
                time.sleep(0.005)  # ~5ms per token

        metrics = DecodeBenchmark.from_performance_recorder(
            recorder, warmup_tokens=2
        )
        assert metrics.total_tokens_generated == 10
        assert metrics.warmup_tokens_excluded == 2
        assert metrics.tokens_per_second > 0.0
        assert metrics.latency_cv >= 0.0

    def test_raises_when_no_decode_events(self) -> None:
        """Verify raises when recorder has no decode token events."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)

        # Only prefill events, no decode
        with recorder.span("prefill", mode="prefill"):
            time.sleep(0.001)

        with pytest.raises(ValueError, match="No 'end_to_end_token_latency'"):
            DecodeBenchmark.from_performance_recorder(recorder)

    def test_gpu_idle_defaults_to_zero(self) -> None:
        """Verify GPU idle is 0 when extracted from recorder (no sysfs data)."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)

        for _ in range(5):
            with recorder.span("end_to_end_token_latency", mode="decode"):
                time.sleep(0.002)

        metrics = DecodeBenchmark.from_performance_recorder(
            recorder, warmup_tokens=0
        )
        assert metrics.gpu_idle_percentage == 0.0


class TestPerformanceTargetValidation:
    """Test performance target validation."""

    def test_all_targets_pass(self) -> None:
        """Verify all targets pass with good metrics."""
        metrics = DecodeMetrics(
            tokens_per_second=15.0,
            gpu_idle_percentage=10.0,
            latency_cv=0.10,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=66.7,
            min_token_latency_ms=55.0,
            max_token_latency_ms=80.0,
            measurement_duration_seconds=3.0,
        )
        results = validate_performance_targets(metrics)
        assert results["tokens_per_second"] is True
        assert results["gpu_idle_percentage"] is True
        assert results["latency_cv"] is True

    def test_throughput_fails(self) -> None:
        """Verify throughput target fails when below 10 tok/s."""
        metrics = DecodeMetrics(
            tokens_per_second=5.0,
            gpu_idle_percentage=10.0,
            latency_cv=0.10,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=200.0,
            min_token_latency_ms=180.0,
            max_token_latency_ms=220.0,
            measurement_duration_seconds=10.0,
        )
        results = validate_performance_targets(metrics)
        assert results["tokens_per_second"] is False
        assert results["gpu_idle_percentage"] is True
        assert results["latency_cv"] is True

    def test_gpu_idle_fails(self) -> None:
        """Verify GPU idle target fails when above 15%."""
        metrics = DecodeMetrics(
            tokens_per_second=12.0,
            gpu_idle_percentage=25.0,
            latency_cv=0.10,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=83.3,
            min_token_latency_ms=70.0,
            max_token_latency_ms=100.0,
            measurement_duration_seconds=4.17,
        )
        results = validate_performance_targets(metrics)
        assert results["tokens_per_second"] is True
        assert results["gpu_idle_percentage"] is False
        assert results["latency_cv"] is True

    def test_latency_cv_fails(self) -> None:
        """Verify latency CV target fails when above 20%."""
        metrics = DecodeMetrics(
            tokens_per_second=12.0,
            gpu_idle_percentage=10.0,
            latency_cv=0.35,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=83.3,
            min_token_latency_ms=40.0,
            max_token_latency_ms=150.0,
            measurement_duration_seconds=4.17,
        )
        results = validate_performance_targets(metrics)
        assert results["tokens_per_second"] is True
        assert results["gpu_idle_percentage"] is True
        assert results["latency_cv"] is False

    def test_custom_targets(self) -> None:
        """Verify custom performance targets work."""
        targets = PerformanceTargets(
            minimum_tokens_per_second=20.0,
            maximum_gpu_idle_percentage=5.0,
            maximum_latency_cv=0.10,
        )
        metrics = DecodeMetrics(
            tokens_per_second=15.0,
            gpu_idle_percentage=3.0,
            latency_cv=0.08,
            total_tokens_generated=50,
            warmup_tokens_excluded=5,
            mean_token_latency_ms=66.7,
            min_token_latency_ms=60.0,
            max_token_latency_ms=75.0,
            measurement_duration_seconds=3.33,
        )
        results = validate_performance_targets(metrics, targets)
        # 15 < 20 minimum
        assert results["tokens_per_second"] is False
        # 3 < 5 maximum
        assert results["gpu_idle_percentage"] is True
        # 0.08 < 0.10 maximum
        assert results["latency_cv"] is True
