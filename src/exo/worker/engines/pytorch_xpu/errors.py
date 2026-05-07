"""
Error handling for PyTorch + IPEX Backend

This module provides custom exception classes and error handling utilities
for the PyTorch + IPEX inference backend.

Requirements addressed:
- 10.1: Device error handling
- 10.2: Model error handling
- 10.4: Inference error handling
"""

from typing import Optional, final


@final
class DeviceError(Exception):
    """
    Exception raised for device-related errors.

    This includes:
    - Device not available
    - Device out of memory
    - Device driver issues
    - Device health check failures

    Requirements: 10.1
    """

    def __init__(
        self,
        message: str,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize DeviceError.

        Args:
            message: Error message
            device_type: Device type string ("xpu", "cuda", or "cpu")
            device_id: Device ID
            original_error: Original exception that caused this error
        """
        self.device_type = device_type
        self.device_id = device_id
        self.original_error = original_error

        full_message = message
        if device_type and device_id is not None:
            full_message = f"[{device_type}:{device_id}] {message}"
        if original_error:
            full_message = f"{full_message} (caused by: {original_error})"

        super().__init__(full_message)


@final
class ModelError(Exception):
    """
    Exception raised for model-related errors.

    This includes:
    - Model not found
    - Model loading failures
    - Model incompatibility
    - Model validation errors

    Requirements: 10.2
    """

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        shard_info: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize ModelError.

        Args:
            message: Error message
            model_id: Model identifier
            shard_info: Shard information (e.g., "layers 0-10")
            original_error: Original exception that caused this error
        """
        self.model_id = model_id
        self.shard_info = shard_info
        self.original_error = original_error

        full_message = message
        if model_id:
            full_message = f"[{model_id}] {message}"
        if shard_info:
            full_message = f"{full_message} ({shard_info})"
        if original_error:
            full_message = f"{full_message} (caused by: {original_error})"

        super().__init__(full_message)


@final
class InferenceError(Exception):
    """
    Exception raised for inference-related errors.

    This includes:
    - Inference execution failures
    - NaN or invalid outputs
    - Timeout errors
    - Numerical instability

    Requirements: 10.4
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        model_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize InferenceError.

        Args:
            message: Error message
            request_id: Request identifier
            model_id: Model identifier
            original_error: Original exception that caused this error
        """
        self.request_id = request_id
        self.model_id = model_id
        self.original_error = original_error

        full_message = message
        if request_id:
            full_message = f"[request:{request_id}] {message}"
        if model_id:
            full_message = f"{full_message} [model:{model_id}]"
        if original_error:
            full_message = f"{full_message} (caused by: {original_error})"

        super().__init__(full_message)


@final
class CacheError(Exception):
    """
    Exception raised for KV cache-related errors.

    This includes:
    - Cache allocation failures
    - Cache eviction errors
    - Cache corruption

    Requirements: 10.4
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        """
        Initialize CacheError.

        Args:
            message: Error message
            request_id: Request identifier
            original_error: Original exception that caused this error
        """
        self.request_id = request_id
        self.original_error = original_error

        full_message = message
        if request_id:
            full_message = f"[request:{request_id}] {message}"
        if original_error:
            full_message = f"{full_message} (caused by: {original_error})"

        super().__init__(full_message)
