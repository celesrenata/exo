"""
Benchmark harness for decode throughput measurement.

Provides instrumentation to measure:
- Tokens per second (sustained decode throughput)
- GPU idle percentage (via sysfs frequency monitoring)
- Per-token latency coefficient of variation (stability metric)

The benchmark integrates with the existing PerformanceRecorder timing hooks
in pipeline_generator.py, which are gated by ``enable_performance_instrumentation``.

**Validates: Requirements 10.1, 10.2, 10.4**
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Final, final

from pydantic import BaseModel, ConfigDict

from exo.worker.engines.pytorch_xpu.instrumentation import (
    PerformanceRecorder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARMUP_TOKENS_DEFAULT: Final[int] = 5
"""Default number of initial tokens to exclude from metrics (warmup period)."""

_GPU_FREQUENCY_SYSFS_PATH: Final[str] = "/sys/class/drm/card0/gt_cur_freq_mhz"
"""Sysfs path for current GPU frequency on Intel iGPUs."""

_GPU_MAX_FREQUENCY_SYSFS_PATH: Final[str] = "/sys/class/drm/card0/gt_max_freq_mhz"
"""Sysfs path for maximum GPU frequency on Intel iGPUs."""

_GPU_IDLE_FREQUENCY_THRESHOLD_MHZ: Final[int] = 300
"""Frequency below which the GPU is considered idle (MHz)."""


# ---------------------------------------------------------------------------
# DecodeMetrics — immutable result container
# ---------------------------------------------------------------------------


@final
class DecodeMetrics(BaseModel):
    """
    Immutable container for decode throughput measurement results.

    All metrics are computed after excluding the warmup period.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    tokens_per_second: float
    """Sustained decode throughput in tokens per second (after warmup)."""

    gpu_idle_percentage: float
    """Percentage of time the GPU was idle during decode (0.0 to 100.0)."""

    latency_cv: float
    """Coefficient of variation of per-token latencies (0.0 to 1.0+).
    Values below 0.20 indicate stable throughput."""

    total_tokens_generated: int
    """Total number of tokens generated (including warmup)."""

    warmup_tokens_excluded: int
    """Number of warmup tokens excluded from metrics."""

    mean_token_latency_ms: float
    """Mean per-token latency in milliseconds (after warmup)."""

    min_token_latency_ms: float
    """Minimum per-token latency in milliseconds (after warmup)."""

    max_token_latency_ms: float
    """Maximum per-token latency in milliseconds (after warmup)."""

    measurement_duration_seconds: float
    """Total measurement duration in seconds (excluding warmup)."""


# ---------------------------------------------------------------------------
# GPU idle measurement via sysfs
# ---------------------------------------------------------------------------


def read_gpu_frequency_mhz(
    sysfs_path: str = _GPU_FREQUENCY_SYSFS_PATH,
) -> int | None:
    """Read the current GPU frequency from sysfs.

    Args:
        sysfs_path: Path to the sysfs file containing current frequency.

    Returns:
        Current GPU frequency in MHz, or None if the file cannot be read.
    """
    try:
        frequency_path = Path(sysfs_path)
        if not frequency_path.exists():
            return None
        content = frequency_path.read_text().strip()
        return int(content)
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read GPU frequency from %s: %s", sysfs_path, exc)
        return None


def read_gpu_max_frequency_mhz(
    sysfs_path: str = _GPU_MAX_FREQUENCY_SYSFS_PATH,
) -> int | None:
    """Read the maximum GPU frequency from sysfs.

    Args:
        sysfs_path: Path to the sysfs file containing maximum frequency.

    Returns:
        Maximum GPU frequency in MHz, or None if the file cannot be read.
    """
    try:
        frequency_path = Path(sysfs_path)
        if not frequency_path.exists():
            return None
        content = frequency_path.read_text().strip()
        return int(content)
    except (OSError, ValueError) as exc:
        logger.debug("Failed to read GPU max frequency from %s: %s", sysfs_path, exc)
        return None


def compute_gpu_idle_percentage(
    frequency_samples: list[int],
    idle_threshold_mhz: int = _GPU_IDLE_FREQUENCY_THRESHOLD_MHZ,
) -> float:
    """Compute GPU idle percentage from frequency samples.

    A sample is considered "idle" when the frequency is at or below the
    idle threshold. This correlates with the GPU having no active workload.

    Args:
        frequency_samples: List of GPU frequency readings in MHz.
        idle_threshold_mhz: Frequency at or below which the GPU is idle.

    Returns:
        Percentage of samples where GPU was idle (0.0 to 100.0).
        Returns 0.0 if no samples are provided.
    """
    if not frequency_samples:
        return 0.0
    idle_count = sum(1 for freq in frequency_samples if freq <= idle_threshold_mhz)
    return (idle_count / len(frequency_samples)) * 100.0


# ---------------------------------------------------------------------------
# Coefficient of variation computation
# ---------------------------------------------------------------------------


def compute_latency_coefficient_of_variation(
    latencies_ms: list[float],
) -> float:
    """Compute the coefficient of variation (CV) of per-token latencies.

    CV = standard_deviation / mean. A CV below 0.20 (20%) indicates stable
    throughput per Requirement 10.4.

    Args:
        latencies_ms: Per-token latencies in milliseconds.

    Returns:
        Coefficient of variation (dimensionless ratio, 0.0 to unbounded).
        Returns 0.0 if fewer than 2 samples or mean is zero.
    """
    count = len(latencies_ms)
    if count < 2:
        return 0.0

    mean = sum(latencies_ms) / count
    if mean == 0.0:
        return 0.0

    variance = sum((latency - mean) ** 2 for latency in latencies_ms) / (count - 1)
    standard_deviation = math.sqrt(variance)
    return standard_deviation / mean


# ---------------------------------------------------------------------------
# DecodeBenchmark — mutable collector for decode performance data
# ---------------------------------------------------------------------------


@final
class DecodeBenchmark:
    """
    Collects per-token timestamps during decode and computes performance metrics.

    Usage:
        benchmark = DecodeBenchmark(warmup_tokens=5)
        benchmark.start()
        for token in decode_loop:
            benchmark.record_token()
            # optionally sample GPU frequency
            freq = read_gpu_frequency_mhz()
            if freq is not None:
                benchmark.record_gpu_frequency(freq)
        metrics = benchmark.compute_metrics()

    The benchmark can also extract timing data from a PerformanceRecorder
    that has been collecting spans during generation.
    """

    __slots__ = (
        "_warmup_tokens",
        "_token_timestamps",
        "_gpu_frequency_samples",
        "_start_time",
        "_idle_threshold_mhz",
    )

    def __init__(
        self,
        warmup_tokens: int = _WARMUP_TOKENS_DEFAULT,
        idle_threshold_mhz: int = _GPU_IDLE_FREQUENCY_THRESHOLD_MHZ,
    ) -> None:
        """Initialize the decode benchmark.

        Args:
            warmup_tokens: Number of initial tokens to exclude from metrics.
            idle_threshold_mhz: GPU frequency threshold for idle detection.
        """
        self._warmup_tokens: int = warmup_tokens
        self._token_timestamps: list[float] = []
        self._gpu_frequency_samples: list[int] = []
        self._start_time: float = 0.0
        self._idle_threshold_mhz: int = idle_threshold_mhz

    @property
    def warmup_tokens(self) -> int:
        """Number of warmup tokens to exclude from metrics."""
        return self._warmup_tokens

    @property
    def token_count(self) -> int:
        """Total number of tokens recorded (including warmup)."""
        return len(self._token_timestamps)

    @property
    def gpu_frequency_sample_count(self) -> int:
        """Number of GPU frequency samples collected."""
        return len(self._gpu_frequency_samples)

    def start(self) -> None:
        """Mark the start of the decode benchmark.

        Call this before the first token is generated.
        """
        self._start_time = time.perf_counter()
        self._token_timestamps.clear()
        self._gpu_frequency_samples.clear()

    def record_token(self) -> None:
        """Record a token generation timestamp.

        Call this immediately after each token is produced.
        """
        self._token_timestamps.append(time.perf_counter())

    def record_gpu_frequency(self, frequency_mhz: int) -> None:
        """Record a GPU frequency sample.

        Args:
            frequency_mhz: Current GPU frequency in MHz.
        """
        self._gpu_frequency_samples.append(frequency_mhz)

    def get_per_token_latencies_ms(self) -> list[float]:
        """Compute per-token latencies in milliseconds.

        The first token's latency is measured from start_time. Subsequent
        tokens are measured from the previous token's timestamp.

        Returns:
            List of per-token latencies in milliseconds.
        """
        if not self._token_timestamps:
            return []

        latencies: list[float] = []
        prev_time = self._start_time

        for timestamp in self._token_timestamps:
            latency_ms = (timestamp - prev_time) * 1000.0
            latencies.append(latency_ms)
            prev_time = timestamp

        return latencies

    def compute_metrics(self) -> DecodeMetrics:
        """Compute decode performance metrics from collected data.

        Excludes the first ``warmup_tokens`` from throughput and stability
        calculations. GPU idle percentage uses all collected frequency samples.

        Returns:
            DecodeMetrics with computed performance values.

        Raises:
            ValueError: If no tokens have been recorded.
        """
        total_tokens = len(self._token_timestamps)
        if total_tokens == 0:
            raise ValueError("No tokens recorded. Call record_token() during decode.")

        all_latencies_ms = self.get_per_token_latencies_ms()

        # Exclude warmup tokens
        effective_warmup = min(self._warmup_tokens, total_tokens - 1)
        post_warmup_latencies = all_latencies_ms[effective_warmup:]

        if not post_warmup_latencies:
            # All tokens were warmup — use all latencies
            post_warmup_latencies = all_latencies_ms
            effective_warmup = 0

        # Compute tokens per second from post-warmup data
        measurement_duration_ms = sum(post_warmup_latencies)
        measurement_duration_seconds = measurement_duration_ms / 1000.0
        post_warmup_token_count = len(post_warmup_latencies)

        tokens_per_second: float
        if measurement_duration_seconds > 0.0:
            tokens_per_second = post_warmup_token_count / measurement_duration_seconds
        else:
            tokens_per_second = 0.0

        # Compute latency statistics
        mean_latency_ms = sum(post_warmup_latencies) / len(post_warmup_latencies)
        min_latency_ms = min(post_warmup_latencies)
        max_latency_ms = max(post_warmup_latencies)

        # Compute coefficient of variation
        latency_cv = compute_latency_coefficient_of_variation(post_warmup_latencies)

        # Compute GPU idle percentage
        gpu_idle_percentage = compute_gpu_idle_percentage(
            self._gpu_frequency_samples,
            idle_threshold_mhz=self._idle_threshold_mhz,
        )

        return DecodeMetrics(
            tokens_per_second=tokens_per_second,
            gpu_idle_percentage=gpu_idle_percentage,
            latency_cv=latency_cv,
            total_tokens_generated=total_tokens,
            warmup_tokens_excluded=effective_warmup,
            mean_token_latency_ms=mean_latency_ms,
            min_token_latency_ms=min_latency_ms,
            max_token_latency_ms=max_latency_ms,
            measurement_duration_seconds=measurement_duration_seconds,
        )

    @staticmethod
    def from_performance_recorder(
        recorder: PerformanceRecorder,
        warmup_tokens: int = _WARMUP_TOKENS_DEFAULT,
        idle_threshold_mhz: int = _GPU_IDLE_FREQUENCY_THRESHOLD_MHZ,
    ) -> DecodeMetrics:
        """Extract decode metrics from a PerformanceRecorder's event data.

        Looks for ``end_to_end_token_latency`` spans in decode mode and
        uses their durations as per-token latencies.

        Args:
            recorder: A PerformanceRecorder that collected spans during generation.
            warmup_tokens: Number of initial decode tokens to exclude.
            idle_threshold_mhz: GPU frequency threshold for idle detection.

        Returns:
            DecodeMetrics computed from the recorder's event data.

        Raises:
            ValueError: If no decode token latency events are found.
        """
        # Extract per-token latencies from end_to_end_token_latency spans
        token_latencies_ms: list[float] = []
        for event in recorder.events:
            if event.event_name == "end_to_end_token_latency" and event.mode == "decode":
                token_latencies_ms.append(event.duration * 1000.0)

        if not token_latencies_ms:
            raise ValueError(
                "No 'end_to_end_token_latency' decode events found in recorder. "
                "Ensure enable_performance_instrumentation is True."
            )

        total_tokens = len(token_latencies_ms)

        # Exclude warmup tokens
        effective_warmup = min(warmup_tokens, total_tokens - 1)
        post_warmup_latencies = token_latencies_ms[effective_warmup:]

        if not post_warmup_latencies:
            post_warmup_latencies = token_latencies_ms
            effective_warmup = 0

        # Compute metrics
        measurement_duration_ms = sum(post_warmup_latencies)
        measurement_duration_seconds = measurement_duration_ms / 1000.0
        post_warmup_token_count = len(post_warmup_latencies)

        tokens_per_second: float
        if measurement_duration_seconds > 0.0:
            tokens_per_second = post_warmup_token_count / measurement_duration_seconds
        else:
            tokens_per_second = 0.0

        mean_latency_ms = sum(post_warmup_latencies) / len(post_warmup_latencies)
        min_latency_ms = min(post_warmup_latencies)
        max_latency_ms = max(post_warmup_latencies)

        latency_cv = compute_latency_coefficient_of_variation(post_warmup_latencies)

        # GPU idle percentage is not available from recorder alone
        gpu_idle_percentage = 0.0

        return DecodeMetrics(
            tokens_per_second=tokens_per_second,
            gpu_idle_percentage=gpu_idle_percentage,
            latency_cv=latency_cv,
            total_tokens_generated=total_tokens,
            warmup_tokens_excluded=effective_warmup,
            mean_token_latency_ms=mean_latency_ms,
            min_token_latency_ms=min_latency_ms,
            max_token_latency_ms=max_latency_ms,
            measurement_duration_seconds=measurement_duration_seconds,
        )


# ---------------------------------------------------------------------------
# Performance target validation
# ---------------------------------------------------------------------------


@final
class PerformanceTargets(BaseModel):
    """Performance targets from Requirements 10.1, 10.2, 10.4."""

    model_config = ConfigDict(frozen=True, strict=True)

    minimum_tokens_per_second: float = 10.0
    """Requirement 10.1: At least 10 tokens per second sustained."""

    maximum_gpu_idle_percentage: float = 15.0
    """Requirement 10.2: GPU idle less than 15%."""

    maximum_latency_cv: float = 0.20
    """Requirement 10.4: CV less than 20% after warmup."""


def validate_performance_targets(
    metrics: DecodeMetrics,
    targets: PerformanceTargets | None = None,
) -> dict[str, bool]:
    """Validate decode metrics against performance targets.

    Args:
        metrics: Computed decode metrics.
        targets: Performance targets to validate against. Uses defaults
            from Requirements 10.1, 10.2, 10.4 if None.

    Returns:
        Dictionary mapping target names to pass/fail booleans.
    """
    if targets is None:
        targets = PerformanceTargets()

    return {
        "tokens_per_second": metrics.tokens_per_second >= targets.minimum_tokens_per_second,
        "gpu_idle_percentage": metrics.gpu_idle_percentage <= targets.maximum_gpu_idle_percentage,
        "latency_cv": metrics.latency_cv <= targets.maximum_latency_cv,
    }
