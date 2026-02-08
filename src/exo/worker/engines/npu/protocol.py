"""Communication protocol for NPU inference service.

This module defines the HTTP/REST API for communicating with the NPU service.
"""

import asyncio
import logging
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

from exo.shared.types.worker.shards import ModelId

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel, frozen=True, strict=True):
    """Request for NPU inference via API."""

    model_id: str
    input_data: dict[str, list[float]]  # Serialized numpy arrays
    request_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    timeout_ms: int = 30000  # 30 second default timeout


class InferenceResponse(BaseModel, frozen=True, strict=True):
    """Response from NPU inference via API."""

    request_id: str
    output_data: dict[str, list[float]]  # Serialized numpy arrays
    inference_time_ms: float
    success: bool
    error_message: str | None = None


class HealthResponse(BaseModel, frozen=True, strict=True):
    """Health check response."""

    status: str  # "healthy" or "unhealthy"
    npu_available: bool
    loaded_models: list[str]
    uptime_seconds: float


class ModelInfo(BaseModel, frozen=True, strict=True):
    """Information about a loaded model."""

    model_id: str
    loaded: bool
    input_shapes: dict[str, list[int]]
    output_shapes: dict[str, list[int]]


class NPUServiceClient:
    """Client for communicating with NPU inference service.

    This client provides a simple interface for sending inference requests
    to the NPU service over HTTP.
    """

    def __init__(
        self, host: str = "localhost", port: int = 52416, timeout: float = 30.0
    ):
        """Initialize NPU service client.

        Args:
            host: Service host
            port: Service port
            timeout: Default request timeout in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "NPUServiceClient":
        """Enter async context manager."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        if self._session:
            await self._session.close()
            self._session = None

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference on NPU service.

        Args:
            request: Inference request

        Returns:
            Inference response

        Raises:
            RuntimeError: If service is unavailable or request fails
        """
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.post(
                f"{self.base_url}/infer",
                json=request.model_dump(),
                timeout=aiohttp.ClientTimeout(total=request.timeout_ms / 1000),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return InferenceResponse(**data)

        except aiohttp.ClientError as e:
            logger.error(f"NPU service request failed: {e}")
            raise RuntimeError(f"NPU service unavailable: {e}") from e
        except asyncio.TimeoutError as e:
            logger.error(f"NPU service request timed out")
            raise RuntimeError("NPU service request timed out") from e

    async def health(self) -> HealthResponse:
        """Check NPU service health.

        Returns:
            Health response

        Raises:
            RuntimeError: If service is unavailable
        """
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                data = await response.json()
                return HealthResponse(**data)

        except aiohttp.ClientError as e:
            logger.error(f"NPU service health check failed: {e}")
            raise RuntimeError(f"NPU service unavailable: {e}") from e

    async def list_models(self) -> list[ModelInfo]:
        """List loaded models on NPU service.

        Returns:
            List of model information

        Raises:
            RuntimeError: If service is unavailable
        """
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.get(f"{self.base_url}/models") as response:
                response.raise_for_status()
                data = await response.json()
                return [ModelInfo(**model) for model in data["models"]]

        except aiohttp.ClientError as e:
            logger.error(f"Failed to list models: {e}")
            raise RuntimeError(f"NPU service unavailable: {e}") from e

    async def load_model(self, model_id: ModelId) -> ModelInfo:
        """Preload a model on NPU service.

        Args:
            model_id: Model identifier

        Returns:
            Model information

        Raises:
            RuntimeError: If model loading fails
        """
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.post(
                f"{self.base_url}/models/{model_id}/load"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return ModelInfo(**data)

        except aiohttp.ClientError as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    async def unload_model(self, model_id: ModelId) -> None:
        """Unload a model from NPU service.

        Args:
            model_id: Model identifier

        Raises:
            RuntimeError: If model unloading fails
        """
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self._session.delete(
                f"{self.base_url}/models/{model_id}"
            ) as response:
                response.raise_for_status()

        except aiohttp.ClientError as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            raise RuntimeError(f"Model unloading failed: {e}") from e


# Server-side API implementation (to be added to service.py)


async def handle_infer(request_data: dict[str, Any], service: Any) -> dict[str, Any]:
    """Handle inference request.

    Args:
        request_data: Request data from HTTP
        service: NPUInferenceService instance

    Returns:
        Response data for HTTP
    """
    from exo.worker.engines.npu.service import NPUInferenceRequest

    # Parse request
    inference_request = InferenceRequest(**request_data)

    # Convert to service request
    service_request = NPUInferenceRequest(
        model_id=ModelId(inference_request.model_id),
        input_data=inference_request.input_data,
        request_id=inference_request.request_id,
    )

    # Execute inference
    response = await service.infer(service_request)

    # Convert to API response
    return InferenceResponse(
        request_id=response.request_id,
        output_data=response.output_data,
        inference_time_ms=response.inference_time_ms,
        success=response.success,
        error_message=response.error_message,
    ).model_dump()


async def handle_health(service: Any) -> dict[str, Any]:
    """Handle health check request.

    Args:
        service: NPUInferenceService instance

    Returns:
        Health response data
    """
    import time

    # Get service status
    npu_available = service.openvino_core is not None
    loaded_models = list(service.loaded_models.keys())

    # Calculate uptime (would need to track start time in service)
    uptime_seconds = 0.0  # Placeholder

    return HealthResponse(
        status="healthy" if npu_available else "unhealthy",
        npu_available=npu_available,
        loaded_models=loaded_models,
        uptime_seconds=uptime_seconds,
    ).model_dump()


async def handle_list_models(service: Any) -> dict[str, Any]:
    """Handle list models request.

    Args:
        service: NPUInferenceService instance

    Returns:
        List of models response data
    """
    models = []

    for model_id, compiled_model in service.loaded_models.items():
        metadata = service.model_metadata.get(model_id, {})

        models.append(
            ModelInfo(
                model_id=model_id,
                loaded=True,
                input_shapes={
                    name: info["shape"]
                    for name, info in metadata.get("inputs", {}).items()
                },
                output_shapes={
                    name: info["shape"]
                    for name, info in metadata.get("outputs", {}).items()
                },
            ).model_dump()
        )

    return {"models": models}


async def handle_load_model(model_id: str, service: Any) -> dict[str, Any]:
    """Handle load model request.

    Args:
        model_id: Model identifier
        service: NPUInferenceService instance

    Returns:
        Model info response data
    """
    # Load model
    await service._load_model(ModelId(model_id))

    # Get metadata
    metadata = service.model_metadata.get(ModelId(model_id), {})

    return ModelInfo(
        model_id=model_id,
        loaded=True,
        input_shapes={
            name: info["shape"] for name, info in metadata.get("inputs", {}).items()
        },
        output_shapes={
            name: info["shape"] for name, info in metadata.get("outputs", {}).items()
        },
    ).model_dump()


async def handle_unload_model(model_id: str, service: Any) -> None:
    """Handle unload model request.

    Args:
        model_id: Model identifier
        service: NPUInferenceService instance
    """
    model_id_typed = ModelId(model_id)

    if model_id_typed in service.loaded_models:
        del service.loaded_models[model_id_typed]
        del service.model_metadata[model_id_typed]
        logger.info(f"Unloaded model {model_id}")
    else:
        logger.warning(f"Model {model_id} not loaded")
