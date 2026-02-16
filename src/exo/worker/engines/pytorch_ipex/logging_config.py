"""
Structured Logging Configuration for PyTorch + IPEX Backend

This module provides structured logging configuration using loguru,
with JSON formatting and appropriate log levels for production use.

Requirements addressed:
- 9.1: Structured logging with loguru
- 9.4: Context-rich log messages
"""

import sys
from typing import Any, Optional

from loguru import logger


def configure_structured_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    include_context: bool = True,
) -> None:
    """
    Configure structured logging for PyTorch+IPEX backend.

    This function sets up loguru with:
    - Appropriate log levels
    - JSON formatting (optional)
    - Context-rich log messages
    - Structured fields for filtering

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to output logs in JSON format
        include_context: Whether to include context fields in logs

    Requirements: 9.1, 9.4

    Example:
        >>> configure_structured_logging(log_level="INFO", json_format=True)
        >>> logger.info("Model loaded", model_id="llama-3.2-3b", device="xpu:0")
    """
    # Remove default handler
    logger.remove()

    # Define log format
    if json_format:
        # JSON format for production/systemd
        log_format = (
            "{{\"timestamp\": \"{time:YYYY-MM-DD HH:mm:ss.SSS}\", "
            "\"level\": \"{level}\", "
            "\"module\": \"{name}\", "
            "\"function\": \"{function}\", "
            "\"line\": {line}, "
            "\"message\": \"{message}\""
        )
        if include_context:
            log_format += ", \"extra\": {extra}"
        log_format += "}}\n"
    else:
        # Human-readable format for development
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        if include_context:
            log_format += " | <yellow>{extra}</yellow>"

    # Add handler with configured format
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=not json_format,
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        "Structured logging configured",
        log_level=log_level,
        json_format=json_format,
        include_context=include_context,
    )


def get_logger_with_context(
    module_name: str,
    **context: Any,
) -> Any:
    """
    Get a logger with pre-bound context fields.

    This creates a logger that automatically includes context fields
    in all log messages, useful for tracking request IDs, device info, etc.

    Args:
        module_name: Name of the module using the logger
        **context: Context fields to bind to the logger

    Returns:
        Logger instance with bound context

    Requirements: 9.1, 9.4

    Example:
        >>> log = get_logger_with_context(
        ...     "pytorch_ipex.inference",
        ...     device="xpu:0",
        ...     backend="pytorch_ipex"
        ... )
        >>> log.info("Starting inference", request_id="req-123")
        # Output includes device="xpu:0", backend="pytorch_ipex", request_id="req-123"
    """
    return logger.bind(module=module_name, **context)


class LogContext:
    """
    Context manager for adding temporary context to logs.

    This allows adding context fields for a specific scope,
    useful for tracking operations like inference requests.

    Requirements: 9.1, 9.4

    Example:
        >>> with LogContext(request_id="req-123", model="llama-3.2-3b"):
        ...     logger.info("Processing request")
        ...     # Log includes request_id and model fields
    """

    def __init__(self, **context: Any) -> None:
        """
        Initialize log context.

        Args:
            **context: Context fields to add to logs
        """
        self.context = context
        self.token: Optional[int] = None

    def __enter__(self) -> "LogContext":
        """Enter context and bind fields to logger."""
        self.token = logger.contextualize(**self.context).__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind fields."""
        if self.token is not None:
            logger.contextualize(**self.context).__exit__(exc_type, exc_val, exc_tb)


def log_device_info(
    device_type: str,
    device_id: int,
    device_name: str,
    total_memory_gb: float,
    free_memory_gb: float,
) -> None:
    """
    Log device information in a structured format.

    Args:
        device_type: Device type (xpu, cuda, cpu)
        device_id: Device ID
        device_name: Device name
        total_memory_gb: Total memory in GB
        free_memory_gb: Free memory in GB

    Requirements: 9.1, 9.4

    Example:
        >>> log_device_info("xpu", 0, "Intel Arc A770", 16.0, 14.5)
    """
    logger.info(
        "Device information",
        device_type=device_type,
        device_id=device_id,
        device_name=device_name,
        total_memory_gb=round(total_memory_gb, 2),
        free_memory_gb=round(free_memory_gb, 2),
        memory_utilization_percent=round(
            (1 - free_memory_gb / total_memory_gb) * 100 if total_memory_gb > 0 else 0,
            2,
        ),
    )


def log_model_info(
    model_id: str,
    shard_info: Optional[str],
    device: str,
    optimization_applied: bool,
    load_time_seconds: float,
) -> None:
    """
    Log model loading information in a structured format.

    Args:
        model_id: Model identifier
        shard_info: Shard information (e.g., "layers 0-10")
        device: Device string (e.g., "xpu:0")
        optimization_applied: Whether IPEX optimization was applied
        load_time_seconds: Time taken to load model

    Requirements: 9.1, 9.4

    Example:
        >>> log_model_info(
        ...     "meta-llama/Llama-3.2-3B",
        ...     "layers 0-28",
        ...     "xpu:0",
        ...     True,
        ...     5.2
        ... )
    """
    logger.info(
        "Model loaded",
        model_id=model_id,
        shard_info=shard_info,
        device=device,
        ipex_optimized=optimization_applied,
        load_time_seconds=round(load_time_seconds, 2),
    )


def log_inference_metrics(
    request_id: str,
    tokens_generated: int,
    duration_seconds: float,
    tokens_per_second: float,
    memory_used_mb: float,
    cache_hit: bool,
) -> None:
    """
    Log inference metrics in a structured format.

    Args:
        request_id: Request identifier
        tokens_generated: Number of tokens generated
        duration_seconds: Inference duration
        tokens_per_second: Throughput
        memory_used_mb: Memory used
        cache_hit: Whether KV cache was used

    Requirements: 9.1, 9.2, 9.4

    Example:
        >>> log_inference_metrics(
        ...     "req-123",
        ...     50,
        ...     2.5,
        ...     20.0,
        ...     1024.0,
        ...     True
        ... )
    """
    logger.info(
        "Inference completed",
        request_id=request_id,
        tokens_generated=tokens_generated,
        duration_seconds=round(duration_seconds, 3),
        tokens_per_second=round(tokens_per_second, 2),
        memory_used_mb=round(memory_used_mb, 2),
        cache_hit=cache_hit,
    )


def log_error_with_context(
    error_type: str,
    error_message: str,
    **context: Any,
) -> None:
    """
    Log an error with structured context.

    Args:
        error_type: Type of error (e.g., "DeviceError", "ModelError")
        error_message: Error message
        **context: Additional context fields

    Requirements: 9.1, 9.4

    Example:
        >>> log_error_with_context(
        ...     "DeviceError",
        ...     "GPU out of memory",
        ...     device="xpu:0",
        ...     request_id="req-123"
        ... )
    """
    logger.error(
        f"{error_type}: {error_message}",
        error_type=error_type,
        **context,
    )
