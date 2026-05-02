"""
Health Check for PyTorch XPU Backend

This module provides health check functionality for monitoring
backend status, device availability, and model loading state.

Requirements addressed:
- 10.5: Health check endpoints
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from loguru import logger


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """
    Result of a health check operation.

    Requirements: 10.5

    Attributes:
        status: Overall health status
        device_available: Whether device is available
        device_healthy: Whether device is functioning properly
        model_loaded: Whether model is loaded
        last_inference_time: Timestamp of last successful inference
        error_message: Error message if unhealthy
        details: Additional health check details
    """

    status: HealthStatus
    device_available: bool
    device_healthy: bool
    model_loaded: bool
    last_inference_time: Optional[float]
    error_message: Optional[str]
    details: dict[str, Any]


class HealthChecker:
    """
    Health checker for PyTorch XPU backend.

    This class provides comprehensive health checking including:
    - Device availability and health
    - Model loading status
    - Recent inference activity
    - Error tracking

    Requirements: 10.5

    Example:
        >>> checker = HealthChecker(device_type="xpu", device_id=0)
        >>> result = checker.check_health()
        >>> print(f"Status: {result.status}")
        >>> print(f"Device healthy: {result.device_healthy}")
    """

    def __init__(
        self,
        device_type: Literal["xpu", "cuda", "cpu"],
        device_id: int,
    ) -> None:
        """
        Initialize health checker.

        Args:
            device_type: Device type (xpu, cuda, cpu)
            device_id: Device ID
        """
        self.device_type = device_type
        self.device_id = device_id
        self.last_inference_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.error_count: int = 0
        self.model_loaded: bool = False
        self.torch_available: bool = False

        # Try to import torch
        try:
            import torch  # type: ignore

            self.torch_available = True
            self.torch = torch
        except ImportError:
            logger.warning("PyTorch not available - health checks will be limited")

    def check_health(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.

        This method checks:
        1. Device availability
        2. Device health (can perform operations)
        3. Model loading status
        4. Recent inference activity
        5. Error conditions

        Returns:
            HealthCheckResult with detailed status

        Requirements: 10.5

        Example:
            >>> result = checker.check_health()
            >>> if result.status == HealthStatus.HEALTHY:
            ...     print("Backend is healthy")
        """
        details: dict[str, Any] = {
            "device_type": self.device_type,
            "device_id": self.device_id,
            "torch_available": self.torch_available,
            "error_count": self.error_count,
        }

        # Check device availability
        device_available = self._check_device_available()
        details["device_available"] = device_available

        # Check device health
        device_healthy = False
        if device_available:
            device_healthy = self._check_device_healthy()
        details["device_healthy"] = device_healthy

        # Check model status
        details["model_loaded"] = self.model_loaded

        # Check recent activity
        if self.last_inference_time is not None:
            time_since_last_inference = time.time() - self.last_inference_time
            details["time_since_last_inference_seconds"] = round(
                time_since_last_inference, 2
            )
        else:
            details["time_since_last_inference_seconds"] = None

        # Determine overall status
        status = self._determine_status(
            device_available=device_available,
            device_healthy=device_healthy,
            model_loaded=self.model_loaded,
        )

        # Error message
        error_message = self.last_error if status == HealthStatus.UNHEALTHY else None

        result = HealthCheckResult(
            status=status,
            device_available=device_available,
            device_healthy=device_healthy,
            model_loaded=self.model_loaded,
            last_inference_time=self.last_inference_time,
            error_message=error_message,
            details=details,
        )

        logger.debug(
            "Health check completed",
            status=status.value,
            device_available=device_available,
            device_healthy=device_healthy,
            model_loaded=self.model_loaded,
        )

        return result

    def _check_device_available(self) -> bool:
        """
        Check if device is available.

        Returns:
            True if device is available, False otherwise
        """
        if not self.torch_available:
            return self.device_type == "cpu"

        try:
            if self.device_type == "xpu":
                if not hasattr(self.torch, "xpu"):
                    return False
                if not self.torch.xpu.is_available():  # type: ignore
                    return False
                device_count: int = self.torch.xpu.device_count()  # type: ignore
                return self.device_id < device_count

            elif self.device_type == "cuda":
                if not self.torch.cuda.is_available():
                    return False
                device_count: int = self.torch.cuda.device_count()
                return self.device_id < device_count

            return self.device_type == "cpu"

        except Exception as e:
            logger.warning(f"Error checking device availability: {e}")
            return False

    def _check_device_healthy(self) -> bool:
        """
        Check if device is healthy by performing a simple operation.

        Returns:
            True if device is healthy, False otherwise
        """
        if not self.torch_available:
            return self.device_type == "cpu"

        try:
            # Try a simple tensor operation
            device = self.torch.device(f"{self.device_type}:{self.device_id}")
            test_tensor = self.torch.tensor([1.0, 2.0, 3.0], device=device)
            result = test_tensor + 1.0
            _ = result.cpu()  # Force synchronization

            return True

        except Exception as e:
            logger.warning(f"Device health check failed: {e}")
            self.last_error = str(e)
            return False

    def _determine_status(
        self,
        device_available: bool,
        device_healthy: bool,
        model_loaded: bool,
    ) -> HealthStatus:
        """
        Determine overall health status based on checks.

        Args:
            device_available: Whether device is available
            device_healthy: Whether device is healthy
            model_loaded: Whether model is loaded

        Returns:
            Overall health status
        """
        # Unhealthy if device not available or not healthy
        if not device_available or not device_healthy:
            return HealthStatus.UNHEALTHY

        # Unhealthy if too many recent errors
        if self.error_count > 10:
            return HealthStatus.UNHEALTHY

        # Degraded if model not loaded
        if not model_loaded:
            return HealthStatus.DEGRADED

        # Degraded if recent errors
        if self.error_count > 0:
            return HealthStatus.DEGRADED

        # Otherwise healthy
        return HealthStatus.HEALTHY

    def record_inference_success(self) -> None:
        """
        Record a successful inference operation.

        This updates the last inference time and resets error count.

        Requirements: 10.5

        Example:
            >>> checker.record_inference_success()
        """
        self.last_inference_time = time.time()
        self.error_count = max(0, self.error_count - 1)  # Decay error count
        self.last_error = None

        logger.debug("Inference success recorded")

    def record_inference_error(self, error: str) -> None:
        """
        Record an inference error.

        This increments error count and stores the error message.

        Requirements: 10.5

        Example:
            >>> checker.record_inference_error("GPU out of memory")
        """
        self.error_count += 1
        self.last_error = error

        logger.debug(
            "Inference error recorded",
            error=error,
            error_count=self.error_count,
        )

    def record_model_loaded(self) -> None:
        """
        Record that a model has been loaded.

        Requirements: 10.5

        Example:
            >>> checker.record_model_loaded()
        """
        self.model_loaded = True
        logger.debug("Model loaded status recorded")

    def record_model_unloaded(self) -> None:
        """
        Record that a model has been unloaded.

        Requirements: 10.5

        Example:
            >>> checker.record_model_unloaded()
        """
        self.model_loaded = False
        logger.debug("Model unloaded status recorded")

    def reset_error_count(self) -> None:
        """
        Reset the error count.

        Useful after recovering from errors or restarting the backend.

        Requirements: 10.5

        Example:
            >>> checker.reset_error_count()
        """
        self.error_count = 0
        self.last_error = None
        logger.debug("Error count reset")

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get a summary of health status.

        Returns:
            Dictionary with health summary

        Requirements: 10.5

        Example:
            >>> summary = checker.get_health_summary()
            >>> print(f"Status: {summary['status']}")
        """
        result = self.check_health()

        return {
            "status": result.status.value,
            "device_available": result.device_available,
            "device_healthy": result.device_healthy,
            "model_loaded": result.model_loaded,
            "last_inference_time": result.last_inference_time,
            "error_message": result.error_message,
            "error_count": self.error_count,
            "details": result.details,
        }


def format_health_response(result: HealthCheckResult) -> dict[str, Any]:
    """
    Format health check result for API response.

    This formats the health check result in a standard format
    suitable for HTTP health check endpoints.

    Args:
        result: Health check result

    Returns:
        Dictionary formatted for API response

    Requirements: 10.5

    Example:
        >>> result = checker.check_health()
        >>> response = format_health_response(result)
        >>> # Returns: {"status": "healthy", "checks": {...}}
    """
    return {
        "status": result.status.value,
        "timestamp": time.time(),
        "checks": {
            "device": {
                "available": result.device_available,
                "healthy": result.device_healthy,
                "type": result.details.get("device_type"),
                "id": result.details.get("device_id"),
            },
            "model": {
                "loaded": result.model_loaded,
            },
            "inference": {
                "last_time": result.last_inference_time,
                "time_since_last": result.details.get(
                    "time_since_last_inference_seconds"
                ),
            },
            "errors": {
                "count": result.details.get("error_count", 0),
                "last_message": result.error_message,
            },
        },
        "details": result.details,
    }
