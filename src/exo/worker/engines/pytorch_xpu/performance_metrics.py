"""
Performance Metrics for PyTorch + IPEX Backend

This module provides performance metrics tracking including:
- Inference latency per request
- GPU utilization monitoring
- Memory usage tracking
- Requests per second counting

Requirements addressed:
- 9.2: Performance metrics tracking
- 9.3: GPU utilization and memory monitoring
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from loguru import logger


@dataclass
class InferenceMetrics:
    """
    Metrics for a single inference operation.

    Requirements: 9.2

    Attributes:
        request_id: Unique request identifier
        tokens_generated: Number of tokens generated
        duration_seconds: Total inference duration
        tokens_per_second: Throughput
        memory_used_mb: Memory used during inference
        gpu_utilization_percent: GPU utilization if available
        cache_hit: Whether KV cache was used
        timestamp: Unix timestamp when metrics were collected
    """

    request_id: str
    tokens_generated: int
    duration_seconds: float
    tokens_per_second: float
    memory_used_mb: float
    gpu_utilization_percent: Optional[float]
    cache_hit: bool
    timestamp: float


@dataclass
class PerformanceMetricsCollector:
    """
    Collects and aggregates performance metrics for PyTorch XPU backend.

    This class tracks:
    - Inference latency per request
    - GPU utilization over time
    - Memory usage patterns
    - Throughput (requests/sec, tokens/sec)

    Requirements: 9.2, 9.3

    Example:
        >>> collector = PerformanceMetricsCollector(
        ...     device_type="xpu",
        ...     device_id=0
        ... )
        >>> collector.record_inference(
        ...     request_id="req-123",
        ...     tokens=50,
        ...     duration=2.5,
        ...     memory_used=1024.0,
        ...     gpu_utilization=75.0,
        ...     cache_hit=True
        ... )
        >>> stats = collector.get_stats()
    """

    device_type: Literal["xpu", "cuda", "cpu"]
    device_id: int
    inference_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    metrics_history: list[InferenceMetrics] = field(default_factory=list)
    max_history_size: int = 100
    start_time: float = field(default_factory=time.time)

    def record_inference(
        self,
        request_id: str,
        tokens: int,
        duration: float,
        memory_used: float,
        gpu_utilization: Optional[float] = None,
        cache_hit: bool = False,
    ) -> InferenceMetrics:
        """
        Record metrics for a single inference operation.

        Args:
            request_id: Unique request identifier
            tokens: Number of tokens generated
            duration: Time taken in seconds
            memory_used: Memory used in MB
            gpu_utilization: GPU utilization percentage (0-100) if available
            cache_hit: Whether KV cache was used

        Returns:
            InferenceMetrics object with the recorded data

        Requirements: 9.2

        Example:
            >>> metrics = collector.record_inference(
            ...     request_id="req-123",
            ...     tokens=50,
            ...     duration=1.0,
            ...     memory_used=512.0,
            ...     gpu_utilization=75.0,
            ...     cache_hit=True
            ... )
        """
        self.inference_count += 1
        self.total_tokens += tokens
        self.total_time += duration

        tokens_per_second = tokens / duration if duration > 0 else 0.0

        metrics = InferenceMetrics(
            request_id=request_id,
            tokens_generated=tokens,
            duration_seconds=duration,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_used,
            gpu_utilization_percent=gpu_utilization,
            cache_hit=cache_hit,
            timestamp=time.time(),
        )

        # Add to history (keep only recent metrics)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

        logger.debug(
            "Inference metrics recorded",
            request_id=request_id,
            device=f"{self.device_type}:{self.device_id}",
            tokens=tokens,
            duration=duration,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_used,
            gpu_utilization_percent=gpu_utilization,
            cache_hit=cache_hit,
        )

        return metrics

    def get_stats(self) -> dict[str, Any]:
        """
        Get aggregated statistics from collected metrics.

        Returns:
            Dictionary containing:
            - inference_count: Total inferences
            - total_tokens: Total tokens generated
            - total_time: Total time spent
            - avg_tokens_per_second: Average throughput
            - recent_tokens_per_second: Recent average (last 10)
            - avg_latency_seconds: Average inference latency
            - p50_latency_seconds: Median latency
            - p95_latency_seconds: 95th percentile latency
            - p99_latency_seconds: 99th percentile latency
            - avg_memory_used_mb: Average memory usage
            - peak_memory_used_mb: Peak memory usage
            - avg_gpu_utilization: Average GPU utilization
            - cache_hit_rate: Percentage of cache hits
            - requests_per_second: Overall request rate

        Requirements: 9.2, 9.3

        Example:
            >>> stats = collector.get_stats()
            >>> print(f"Avg throughput: {stats['avg_tokens_per_second']:.2f} tok/s")
            >>> print(f"P95 latency: {stats['p95_latency_seconds']:.3f}s")
        """
        if self.inference_count == 0:
            return {
                "inference_count": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "avg_tokens_per_second": 0.0,
                "recent_tokens_per_second": 0.0,
                "avg_latency_seconds": 0.0,
                "p50_latency_seconds": 0.0,
                "p95_latency_seconds": 0.0,
                "p99_latency_seconds": 0.0,
                "avg_memory_used_mb": 0.0,
                "peak_memory_used_mb": 0.0,
                "avg_gpu_utilization": None,
                "cache_hit_rate": 0.0,
                "requests_per_second": 0.0,
            }

        # Overall averages
        avg_tokens_per_second = (
            self.total_tokens / self.total_time if self.total_time > 0 else 0.0
        )

        # Recent averages (last 10 inferences)
        recent_metrics = self.metrics_history[-10:]
        recent_tokens_per_second = (
            sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
            if recent_metrics
            else 0.0
        )

        # Latency statistics
        latencies = [m.duration_seconds for m in self.metrics_history]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Percentile calculations
        sorted_latencies = sorted(latencies)
        p50_latency = self._percentile(sorted_latencies, 50)
        p95_latency = self._percentile(sorted_latencies, 95)
        p99_latency = self._percentile(sorted_latencies, 99)

        # Memory statistics
        memory_values = [m.memory_used_mb for m in self.metrics_history]
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0.0
        peak_memory = max(memory_values) if memory_values else 0.0

        # GPU utilization (only if available)
        gpu_metrics = [
            m.gpu_utilization_percent
            for m in self.metrics_history
            if m.gpu_utilization_percent is not None
        ]
        avg_gpu_utilization = (
            sum(gpu_metrics) / len(gpu_metrics) if gpu_metrics else None
        )

        # Cache hit rate
        cache_hits = sum(1 for m in self.metrics_history if m.cache_hit)
        cache_hit_rate = (
            (cache_hits / len(self.metrics_history)) * 100
            if self.metrics_history
            else 0.0
        )

        # Requests per second
        elapsed_time = time.time() - self.start_time
        requests_per_second = (
            self.inference_count / elapsed_time if elapsed_time > 0 else 0.0
        )

        return {
            "inference_count": self.inference_count,
            "total_tokens": self.total_tokens,
            "total_time": round(self.total_time, 2),
            "avg_tokens_per_second": round(avg_tokens_per_second, 2),
            "recent_tokens_per_second": round(recent_tokens_per_second, 2),
            "avg_latency_seconds": round(avg_latency, 3),
            "p50_latency_seconds": round(p50_latency, 3),
            "p95_latency_seconds": round(p95_latency, 3),
            "p99_latency_seconds": round(p99_latency, 3),
            "avg_memory_used_mb": round(avg_memory, 2),
            "peak_memory_used_mb": round(peak_memory, 2),
            "avg_gpu_utilization": (
                round(avg_gpu_utilization, 2) if avg_gpu_utilization is not None else None
            ),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "requests_per_second": round(requests_per_second, 2),
        }

    def _percentile(self, sorted_values: list[float], percentile: int) -> float:
        """
        Calculate percentile from sorted values.

        Args:
            sorted_values: List of values sorted in ascending order
            percentile: Percentile to calculate (0-100)

        Returns:
            Value at the specified percentile
        """
        if not sorted_values:
            return 0.0

        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_recent_metrics(self, count: int = 10) -> list[InferenceMetrics]:
        """
        Get the most recent inference metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent InferenceMetrics objects

        Requirements: 9.2

        Example:
            >>> recent = collector.get_recent_metrics(5)
            >>> for m in recent:
            ...     print(f"{m.request_id}: {m.tokens_per_second:.2f} tok/s")
        """
        return self.metrics_history[-count:]

    def reset(self) -> None:
        """
        Reset all collected metrics.

        Useful when starting a new inference session or after
        a backend reconfiguration.

        Requirements: 9.2

        Example:
            >>> collector.reset()
            >>> assert collector.inference_count == 0
        """
        self.inference_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.metrics_history.clear()
        self.start_time = time.time()

        logger.debug(
            "Metrics collector reset",
            device=f"{self.device_type}:{self.device_id}",
        )


def collect_gpu_metrics(
    device_type: Literal["xpu", "cuda", "cpu"],
    device_id: int,
) -> dict[str, Any]:
    """
    Collect current GPU metrics (utilization, memory, temperature).

    This function queries the GPU for current performance metrics.
    For Intel Arc (XPU), it uses torch.xpu APIs.
    For NVIDIA (CUDA), it uses torch.cuda APIs.
    For CPU, it returns minimal metrics.

    Args:
        device_type: Device type (xpu, cuda, cpu)
        device_id: Device ID

    Returns:
        Dictionary containing:
        - memory_used_mb: Current memory usage
        - memory_total_mb: Total memory available
        - memory_utilization_percent: Memory utilization percentage
        - gpu_utilization_percent: GPU utilization (if available)
        - temperature_celsius: GPU temperature (if available)

    Requirements: 9.2, 9.3

    Example:
        >>> metrics = collect_gpu_metrics("xpu", 0)
        >>> print(f"GPU utilization: {metrics['gpu_utilization_percent']}%")
    """
    metrics: dict[str, Any] = {
        "memory_used_mb": 0.0,
        "memory_total_mb": 0.0,
        "memory_utilization_percent": 0.0,
        "gpu_utilization_percent": None,
        "temperature_celsius": None,
    }

    try:
        import torch  # type: ignore

        if device_type == "xpu" and hasattr(torch, "xpu"):
            # Intel Arc GPU metrics
            if torch.xpu.is_available():  # type: ignore
                props = torch.xpu.get_device_properties(device_id)  # type: ignore
                allocated = torch.xpu.memory_allocated(device_id)  # type: ignore
                total: int = props.total_memory  # type: ignore

                metrics["memory_used_mb"] = allocated / (1024 * 1024)
                metrics["memory_total_mb"] = total / (1024 * 1024)
                metrics["memory_utilization_percent"] = (
                    (allocated / total) * 100 if total > 0 else 0.0
                )

                # Note: GPU utilization and temperature not directly available
                # via torch.xpu API. Would need Level Zero or OpenCL queries.
                logger.debug(
                    "XPU metrics collected",
                    device_id=device_id,
                    memory_used_mb=metrics["memory_used_mb"],
                    memory_total_mb=metrics["memory_total_mb"],
                )

        elif device_type == "cuda" and torch.cuda.is_available():
            # NVIDIA GPU metrics
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            total: int = props.total_memory

            metrics["memory_used_mb"] = allocated / (1024 * 1024)
            metrics["memory_total_mb"] = total / (1024 * 1024)
            metrics["memory_utilization_percent"] = (
                (allocated / total) * 100 if total > 0 else 0.0
            )

            # Try to get GPU utilization via nvidia-smi (if available)
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                        f"--id={device_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        metrics["gpu_utilization_percent"] = float(parts[0].strip())
                        metrics["temperature_celsius"] = float(parts[1].strip())
            except Exception as e:
                logger.debug(f"Could not query nvidia-smi: {e}")

            logger.debug(
                "CUDA metrics collected",
                device_id=device_id,
                memory_used_mb=metrics["memory_used_mb"],
                memory_total_mb=metrics["memory_total_mb"],
                gpu_utilization_percent=metrics["gpu_utilization_percent"],
            )

        elif device_type == "cpu":
            # CPU has no meaningful GPU metrics
            logger.debug("CPU device - no GPU metrics available")

    except Exception as e:
        logger.warning(
            f"Failed to collect GPU metrics for {device_type}:{device_id}: {e}"
        )

    return metrics
