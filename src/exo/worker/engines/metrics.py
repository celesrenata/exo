"""Backend performance metrics collection.

This module provides metrics collection for inference backends,
tracking performance indicators like tokens/sec, memory usage,
and GPU utilization.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from exo.shared.types.worker.runners import RunnerId


@dataclass
class InferenceMetrics:
    """Metrics collected during a single inference operation.

    Attributes:
        backend_type: Type of backend (mlx, tinygrad, npu)
        device_type: Device used (CPU, GPU, NPU, METAL)
        tokens_per_second: Inference throughput
        memory_used_mb: Memory used during inference
        gpu_utilization_percent: GPU utilization if available
        timestamp: Unix timestamp when metrics were collected
    """

    backend_type: str
    device_type: str
    tokens_per_second: float
    memory_used_mb: float
    gpu_utilization_percent: float | None
    timestamp: float


@dataclass
class BackendMetricsCollector:
    """Collects and aggregates performance metrics for a backend.

    This class tracks inference metrics over time and can emit
    events to the cluster state for dashboard visibility.

    Attributes:
        runner_id: ID of the runner this collector is tracking
        backend_type: Type of backend (mlx, tinygrad, npu)
        device_type: Device type (CPU, GPU, NPU, METAL)
        inference_count: Total number of inferences performed
        total_tokens: Total tokens generated
        total_time: Total inference time in seconds
        metrics_history: Recent metrics for averaging

    Example:
        >>> collector = BackendMetricsCollector(
        ...     runner_id=RunnerId("runner-1"),
        ...     backend_type="tinygrad",
        ...     device_type="GPU"
        ... )
        >>> collector.record_inference(
        ...     tokens=100,
        ...     duration=2.5,
        ...     memory_used=1024.0
        ... )
        >>> stats = collector.get_stats()
        >>> print(f"Avg tokens/sec: {stats['avg_tokens_per_second']:.2f}")
    """

    runner_id: RunnerId
    backend_type: str
    device_type: str
    inference_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    metrics_history: list[InferenceMetrics] = field(default_factory=list)
    max_history_size: int = 100

    def record_inference(
        self,
        tokens: int,
        duration: float,
        memory_used: float,
        gpu_utilization: float | None = None,
    ) -> InferenceMetrics:
        """Record metrics for a single inference operation.

        Args:
            tokens: Number of tokens generated
            duration: Time taken in seconds
            memory_used: Memory used in MB
            gpu_utilization: GPU utilization percentage (0-100) if available

        Returns:
            InferenceMetrics object with the recorded data

        Example:
            >>> metrics = collector.record_inference(
            ...     tokens=50,
            ...     duration=1.0,
            ...     memory_used=512.0,
            ...     gpu_utilization=75.0
            ... )
        """
        self.inference_count += 1
        self.total_tokens += tokens
        self.total_time += duration

        tokens_per_second = tokens / duration if duration > 0 else 0.0

        metrics = InferenceMetrics(
            backend_type=self.backend_type,
            device_type=self.device_type,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_used,
            gpu_utilization_percent=gpu_utilization,
            timestamp=time.time(),
        )

        # Add to history (keep only recent metrics)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

        logger.debug(
            "Inference metrics recorded",
            runner_id=str(self.runner_id),
            backend_type=self.backend_type,
            device_type=self.device_type,
            tokens=tokens,
            duration=duration,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_used,
            gpu_utilization_percent=gpu_utilization,
        )

        return metrics

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics from collected metrics.

        Returns:
            Dictionary containing aggregated statistics including:
            - inference_count: Total inferences
            - total_tokens: Total tokens generated
            - total_time: Total time spent
            - avg_tokens_per_second: Average throughput
            - recent_tokens_per_second: Recent average (last 10 inferences)
            - avg_memory_used_mb: Average memory usage
            - avg_gpu_utilization: Average GPU utilization if available

        Example:
            >>> stats = collector.get_stats()
            >>> print(f"Total inferences: {stats['inference_count']}")
            >>> print(f"Avg throughput: {stats['avg_tokens_per_second']:.2f} tok/s")
        """
        if self.inference_count == 0:
            return {
                "inference_count": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "avg_tokens_per_second": 0.0,
                "recent_tokens_per_second": 0.0,
                "avg_memory_used_mb": 0.0,
                "avg_gpu_utilization": None,
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

        avg_memory_used_mb = (
            sum(m.memory_used_mb for m in self.metrics_history)
            / len(self.metrics_history)
            if self.metrics_history
            else 0.0
        )

        # GPU utilization (only if available)
        gpu_metrics = [
            m.gpu_utilization_percent
            for m in self.metrics_history
            if m.gpu_utilization_percent is not None
        ]
        avg_gpu_utilization = (
            sum(gpu_metrics) / len(gpu_metrics) if gpu_metrics else None
        )

        return {
            "inference_count": self.inference_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "recent_tokens_per_second": recent_tokens_per_second,
            "avg_memory_used_mb": avg_memory_used_mb,
            "avg_gpu_utilization": avg_gpu_utilization,
        }

    def emit_metrics_event(self, event_emitter: Any) -> None:
        """Emit current metrics as an event to cluster state.

        This allows the dashboard to display real-time performance metrics.

        Args:
            event_emitter: Event emitter to send metrics events

        Example:
            >>> collector.emit_metrics_event(worker.event_emitter)
        """
        if not self.metrics_history:
            return

        # Get most recent metrics
        latest = self.metrics_history[-1]

        # Import here to avoid circular dependency
        from exo.shared.types.events import GPUMetricsCollected

        # Only emit GPU metrics if we have GPU utilization data
        if latest.gpu_utilization_percent is not None:
            event = GPUMetricsCollected(
                runner_id=self.runner_id,
                device_name=f"{self.device_type} ({self.backend_type})",
                runtime=self.backend_type,
                memory_used_mb=latest.memory_used_mb,
                memory_total_mb=0.0,  # TODO: Track total memory
                utilization_percent=latest.gpu_utilization_percent,
                timestamp=latest.timestamp,
            )

            try:
                event_emitter.emit(event)
                logger.debug(
                    "GPU metrics event emitted",
                    runner_id=str(self.runner_id),
                    backend_type=self.backend_type,
                    device_type=self.device_type,
                )
            except Exception as e:
                logger.warning(
                    "Failed to emit GPU metrics event",
                    runner_id=str(self.runner_id),
                    error=str(e),
                )

    def reset(self) -> None:
        """Reset all collected metrics.

        Useful when starting a new inference session or after
        a backend reconfiguration.

        Example:
            >>> collector.reset()
            >>> assert collector.inference_count == 0
        """
        self.inference_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.metrics_history.clear()

        logger.debug(
            "Metrics collector reset",
            runner_id=str(self.runner_id),
            backend_type=self.backend_type,
        )
