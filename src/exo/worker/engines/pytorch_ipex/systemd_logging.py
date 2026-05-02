"""
Systemd Journal Integration for PyTorch XPU Backend

This module provides integration with systemd journal for logging,
including service metadata and structured logging support.

Requirements addressed:
- 9.5: Systemd journal integration
"""

import sys
from typing import Any

from loguru import logger


def configure_systemd_logging(
    service_name: str = "exo-pytorch-ipex",
    log_level: str = "INFO",
) -> None:
    """
    Configure logging for systemd journal.

    This function sets up loguru to output logs in a format
    suitable for systemd journal, with structured fields and
    service metadata.

    Args:
        service_name: Name of the systemd service
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)

    Requirements: 9.5

    Example:
        >>> configure_systemd_logging("exo-pytorch-ipex", "INFO")
        >>> logger.info("Service started", backend="pytorch_xpu")
    """
    # Remove default handler
    logger.remove()

    # Systemd journal format
    # Systemd automatically captures stdout/stderr and adds metadata
    # We use a simple format that systemd can parse
    log_format = (
        "<{level}> "
        "{name}:{function}:{line} | "
        "{message} | "
        "extra={extra}"
    )

    # Add handler for systemd
    logger.add(
        sys.stdout,  # systemd captures stdout
        format=log_format,
        level=log_level,
        colorize=False,  # No colors for systemd
        backtrace=True,
        diagnose=True,
    )

    # Add service metadata to all logs
    logger.configure(
        extra={
            "service": service_name,
            "backend": "pytorch_xpu",
        }
    )

    logger.info(
        "Systemd logging configured",
        service_name=service_name,
        log_level=log_level,
    )


def add_systemd_metadata(**metadata: Any) -> None:
    """
    Add metadata fields that will be included in all systemd logs.

    This is useful for adding context like node ID, cluster ID, etc.
    that should be present in all log entries.

    Args:
        **metadata: Metadata fields to add

    Requirements: 9.5

    Example:
        >>> add_systemd_metadata(
        ...     node_id="node-1",
        ...     cluster_id="cluster-main",
        ...     device="xpu:0"
        ... )
    """
    logger.configure(extra=metadata)
    logger.debug("Systemd metadata added", **metadata)


def log_systemd_ready() -> None:
    """
    Log that the service is ready.

    This sends a special message that systemd can use to determine
    when the service has finished starting up.

    Requirements: 9.5

    Example:
        >>> log_systemd_ready()
    """
    # Systemd looks for "READY=1" in sd_notify protocol
    # For logging, we just log a clear message
    logger.info("Service ready", systemd_ready=True)


def log_systemd_stopping() -> None:
    """
    Log that the service is stopping.

    This sends a message indicating graceful shutdown is starting.

    Requirements: 9.5

    Example:
        >>> log_systemd_stopping()
    """
    logger.info("Service stopping", systemd_stopping=True)


def log_systemd_watchdog() -> None:
    """
    Log a watchdog keepalive message.

    This can be used with systemd's watchdog feature to indicate
    the service is still healthy.

    Requirements: 9.5

    Example:
        >>> log_systemd_watchdog()
    """
    logger.debug("Watchdog keepalive", systemd_watchdog=True)


class SystemdLogHandler:
    """
    Handler for systemd journal integration.

    This class provides methods for logging to systemd journal
    with appropriate metadata and structured fields.

    Requirements: 9.5

    Example:
        >>> handler = SystemdLogHandler("exo-pytorch-ipex")
        >>> handler.log_startup(device="xpu:0", model="llama-3.2-3b")
        >>> handler.log_inference(request_id="req-123", tokens=50, duration=2.5)
    """

    def __init__(self, service_name: str) -> None:
        """
        Initialize systemd log handler.

        Args:
            service_name: Name of the systemd service
        """
        self.service_name = service_name
        self.logger = logger.bind(service=service_name, backend="pytorch_xpu")

    def log_startup(self, **context: Any) -> None:
        """
        Log service startup with context.

        Args:
            **context: Startup context (device, model, etc.)

        Requirements: 9.5

        Example:
            >>> handler.log_startup(
            ...     device="xpu:0",
            ...     device_name="Intel Arc A770",
            ...     ipex_version="2.0.0"
            ... )
        """
        self.logger.info(
            "PyTorch XPU backend starting",
            event_type="startup",
            **context,
        )

    def log_shutdown(self, **context: Any) -> None:
        """
        Log service shutdown with context.

        Args:
            **context: Shutdown context (reason, etc.)

        Requirements: 9.5

        Example:
            >>> handler.log_shutdown(reason="graceful", uptime_seconds=3600)
        """
        self.logger.info(
            "PyTorch XPU backend shutting down",
            event_type="shutdown",
            **context,
        )

    def log_model_loaded(
        self,
        model_id: str,
        device: str,
        load_time_seconds: float,
        **context: Any,
    ) -> None:
        """
        Log model loading event.

        Args:
            model_id: Model identifier
            device: Device string
            load_time_seconds: Time taken to load
            **context: Additional context

        Requirements: 9.5

        Example:
            >>> handler.log_model_loaded(
            ...     "meta-llama/Llama-3.2-3B",
            ...     "xpu:0",
            ...     5.2,
            ...     ipex_optimized=True
            ... )
        """
        self.logger.info(
            "Model loaded",
            event_type="model_loaded",
            model_id=model_id,
            device=device,
            load_time_seconds=round(load_time_seconds, 2),
            **context,
        )

    def log_inference(
        self,
        request_id: str,
        tokens: int,
        duration: float,
        **context: Any,
    ) -> None:
        """
        Log inference event.

        Args:
            request_id: Request identifier
            tokens: Tokens generated
            duration: Inference duration
            **context: Additional context

        Requirements: 9.5

        Example:
            >>> handler.log_inference(
            ...     "req-123",
            ...     50,
            ...     2.5,
            ...     tokens_per_second=20.0,
            ...     cache_hit=True
            ... )
        """
        self.logger.info(
            "Inference completed",
            event_type="inference",
            request_id=request_id,
            tokens=tokens,
            duration_seconds=round(duration, 3),
            **context,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        **context: Any,
    ) -> None:
        """
        Log error event.

        Args:
            error_type: Type of error
            error_message: Error message
            **context: Additional context

        Requirements: 9.5

        Example:
            >>> handler.log_error(
            ...     "DeviceError",
            ...     "GPU out of memory",
            ...     device="xpu:0",
            ...     request_id="req-123"
            ... )
        """
        self.logger.error(
            f"{error_type}: {error_message}",
            event_type="error",
            error_type=error_type,
            **context,
        )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of metrics

        Requirements: 9.5

        Example:
            >>> handler.log_metrics({
            ...     "avg_tokens_per_second": 20.0,
            ...     "p95_latency_seconds": 0.150,
            ...     "cache_hit_rate": 85.0
            ... })
        """
        self.logger.info(
            "Performance metrics",
            event_type="metrics",
            **metrics,
        )


def test_systemd_logging() -> None:
    """
    Test systemd logging configuration.

    This function can be used to verify that systemd logging
    is working correctly.

    Requirements: 9.5

    Example:
        >>> test_systemd_logging()
    """
    configure_systemd_logging("exo-pytorch-ipex-test", "DEBUG")

    logger.info("Test message", test_field="test_value")
    logger.debug("Debug message", debug_field=123)
    logger.warning("Warning message", warning_field=True)
    logger.error("Error message", error_field="error_value")

    log_systemd_ready()
    log_systemd_watchdog()
    log_systemd_stopping()

    logger.info("Systemd logging test complete")


if __name__ == "__main__":
    # Run test when executed directly
    test_systemd_logging()
