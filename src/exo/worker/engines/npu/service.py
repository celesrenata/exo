"""Intel NPU inference service using OpenVINO.

This module implements a standalone inference service that executes models
on Intel NPU hardware using OpenVINO. It runs as a separate process/service
and communicates with exo workers via HTTP API.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exo.shared.types.worker.shards import ModelId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NPUInferenceRequest:
    """Request for NPU inference."""

    model_id: ModelId
    input_data: dict[str, Any]  # Input tensors as numpy arrays (serialized)
    request_id: str


@dataclass(frozen=True)
class NPUInferenceResponse:
    """Response from NPU inference."""

    request_id: str
    output_data: dict[str, Any]  # Output tensors as numpy arrays (serialized)
    inference_time_ms: float
    success: bool
    error_message: str | None = None


class NPUInferenceService:
    """Standalone service for NPU inference using OpenVINO.

    This service:
    1. Initializes OpenVINO core and configures for NPU device
    2. Loads and caches models
    3. Executes inference requests on NPU
    4. Provides HTTP API for communication with exo workers
    """

    def __init__(self, port: int = 52416, cache_dir: str | None = None):
        """Initialize NPU inference service.

        Args:
            port: Port to listen on for HTTP API
            cache_dir: Directory to cache compiled models
        """
        self.port = port
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "exo" / "npu"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.openvino_core: Any | None = None
        self.loaded_models: dict[ModelId, Any] = {}  # ModelId -> compiled model
        self.model_metadata: dict[ModelId, dict[str, Any]] = {}  # Model input/output info

        self._running = False
        self._server: Any | None = None
        self._app: Any | None = None
        self._server_task: Any | None = None

    async def start(self) -> None:
        """Start NPU inference service.

        Initializes OpenVINO and starts HTTP server.
        """
        logger.info(f"Starting NPU inference service on port {self.port}")

        # Initialize OpenVINO
        try:
            import openvino as ov

            self.openvino_core = ov.Core()

            # Verify NPU device is available
            available_devices = self.openvino_core.available_devices()
            if not any("NPU" in device for device in available_devices):
                raise RuntimeError(f"NPU device not found. Available devices: {available_devices}")

            logger.info(f"OpenVINO initialized. Available devices: {available_devices}")

        except ImportError as e:
            logger.error("OpenVINO not installed. Cannot start NPU service.")
            raise RuntimeError("OpenVINO not available") from e
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO: {e}")
            raise

        # Start HTTP server
        await self._start_server()

        self._running = True
        logger.info("NPU inference service started successfully")

    async def stop(self) -> None:
        """Stop NPU inference service and clean up resources."""
        logger.info("Stopping NPU inference service")
        self._running = False

        # Stop HTTP server
        if self._server:
            await self._stop_server()

        # Unload all models
        self.loaded_models.clear()
        self.model_metadata.clear()

        logger.info("NPU inference service stopped")

    async def infer(self, request: NPUInferenceRequest) -> NPUInferenceResponse:
        """Execute inference on NPU.

        Args:
            request: Inference request with model ID and input data

        Returns:
            Inference response with output data
        """
        import time

        start_time = time.time()

        try:
            # Load model if not cached
            if request.model_id not in self.loaded_models:
                logger.info(f"Loading model {request.model_id} for first use")
                await self._load_model(request.model_id)

            compiled_model = self.loaded_models[request.model_id]
            metadata = self.model_metadata[request.model_id]

            # Prepare input tensors
            input_tensors = self._prepare_inputs(request.input_data, metadata)

            # Execute inference on NPU
            logger.debug(f"Executing inference for {request.model_id} on NPU")
            infer_request = compiled_model.create_infer_request()
            infer_request.infer(input_tensors)

            # Extract output tensors
            output_data = self._extract_outputs(infer_request, metadata)

            inference_time_ms = (time.time() - start_time) * 1000

            logger.debug(f"Inference completed in {inference_time_ms:.2f}ms")

            return NPUInferenceResponse(
                request_id=request.request_id,
                output_data=output_data,
                inference_time_ms=inference_time_ms,
                success=True,
            )

        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Inference failed for {request.model_id}: {e}")

            return NPUInferenceResponse(
                request_id=request.request_id,
                output_data={},
                inference_time_ms=inference_time_ms,
                success=False,
                error_message=str(e),
            )

    async def _load_model(self, model_id: ModelId) -> None:
        """Load and compile model for NPU execution.

        Args:
            model_id: Model identifier

        Raises:
            RuntimeError: If model loading fails
        """
        if self.openvino_core is None:
            raise RuntimeError("OpenVINO not initialized")

        logger.info(f"Loading model {model_id} for NPU")

        try:
            # Determine model path
            # For now, assume models are in standard HuggingFace cache
            # In production, this would integrate with exo's model download system
            model_path = self._resolve_model_path(model_id)

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Read model
            logger.debug(f"Reading model from {model_path}")
            model = self.openvino_core.read_model(model_path)

            # Compile for NPU
            logger.debug(f"Compiling model for NPU device")
            compiled_model = self.openvino_core.compile_model(model, "NPU")

            # Cache compiled model
            self.loaded_models[model_id] = compiled_model

            # Store metadata about inputs/outputs
            self.model_metadata[model_id] = {
                "inputs": {inp.any_name: {"shape": inp.shape, "dtype": inp.element_type.to_dtype()} for inp in model.inputs},
                "outputs": {out.any_name: {"shape": out.shape, "dtype": out.element_type.to_dtype()} for out in model.outputs},
            }

            logger.info(f"Model {model_id} loaded and compiled for NPU")

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _resolve_model_path(self, model_id: ModelId) -> Path:
        """Resolve model ID to file path.

        Args:
            model_id: Model identifier

        Returns:
            Path to model file (OpenVINO IR format)
        """
        # Check cache directory first
        cached_model = self.cache_dir / f"{model_id}.xml"
        if cached_model.exists():
            return cached_model

        # Check HuggingFace cache
        # This is a simplified implementation - production would integrate
        # with exo's download system
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dirs = list(hf_cache.glob(f"models--{model_id.replace('/', '--')}"))

        if model_dirs:
            # Look for OpenVINO IR files
            for model_dir in model_dirs:
                ir_files = list(model_dir.rglob("*.xml"))
                if ir_files:
                    return ir_files[0]

        raise FileNotFoundError(f"Model {model_id} not found in cache")

    def _prepare_inputs(self, input_data: dict[str, Any], metadata: dict[str, Any]) -> dict[str, np.ndarray]:
        """Prepare input tensors for inference.

        Args:
            input_data: Input data (may be serialized)
            metadata: Model metadata with input specifications

        Returns:
            Dictionary of input tensors
        """
        input_tensors = {}

        for name, data in input_data.items():
            # Convert to numpy array if needed
            if isinstance(data, list):
                data = np.array(data)
            elif not isinstance(data, np.ndarray):
                raise ValueError(f"Input {name} must be numpy array or list")

            # Validate shape and dtype match metadata
            expected_dtype = metadata["inputs"][name]["dtype"]
            if data.dtype != expected_dtype:
                logger.debug(f"Converting input {name} from {data.dtype} to {expected_dtype}")
                data = data.astype(expected_dtype)

            input_tensors[name] = data

        return input_tensors

    def _extract_outputs(self, infer_request: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract output tensors from inference request.

        Args:
            infer_request: OpenVINO inference request
            metadata: Model metadata with output specifications

        Returns:
            Dictionary of output tensors (as lists for JSON serialization)
        """
        output_data = {}

        for name in metadata["outputs"].keys():
            tensor = infer_request.get_output_tensor(name)
            # Convert to list for JSON serialization
            output_data[name] = tensor.data.tolist()

        return output_data

    async def _start_server(self) -> None:
        """Start HTTP server for API using FastAPI."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            import uvicorn

            app = FastAPI(title="Intel NPU Inference Service")

            # Store service reference for handlers
            app.state.npu_service = self

            @app.post("/infer")
            async def infer_endpoint(request_data: dict[str, Any]) -> JSONResponse:
                """Execute inference on NPU."""
                from exo.worker.engines.npu.protocol import handle_infer

                try:
                    response = await handle_infer(request_data, app.state.npu_service)
                    return JSONResponse(content=response)
                except Exception as e:
                    logger.error(f"Inference request failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/health")
            async def health_endpoint() -> JSONResponse:
                """Health check endpoint."""
                from exo.worker.engines.npu.protocol import handle_health

                try:
                    response = await handle_health(app.state.npu_service)
                    return JSONResponse(content=response)
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/models")
            async def list_models_endpoint() -> JSONResponse:
                """List loaded models."""
                from exo.worker.engines.npu.protocol import handle_list_models

                try:
                    response = await handle_list_models(app.state.npu_service)
                    return JSONResponse(content=response)
                except Exception as e:
                    logger.error(f"List models failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/models/{model_id}/load")
            async def load_model_endpoint(model_id: str) -> JSONResponse:
                """Preload a model."""
                from exo.worker.engines.npu.protocol import handle_load_model

                try:
                    response = await handle_load_model(model_id, app.state.npu_service)
                    return JSONResponse(content=response)
                except Exception as e:
                    logger.error(f"Load model failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.delete("/models/{model_id}")
            async def unload_model_endpoint(model_id: str) -> JSONResponse:
                """Unload a model."""
                from exo.worker.engines.npu.protocol import handle_unload_model

                try:
                    await handle_unload_model(model_id, app.state.npu_service)
                    return JSONResponse(content={"status": "success"})
                except Exception as e:
                    logger.error(f"Unload model failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            # Store app and start server in background
            self._app = app
            config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="info")
            self._server = uvicorn.Server(config)

            # Start server in background task
            self._server_task = asyncio.create_task(self._server.serve())

            logger.info(f"HTTP server started on port {self.port}")

        except ImportError as e:
            logger.error("FastAPI or uvicorn not installed. Cannot start HTTP server.")
            raise RuntimeError("FastAPI/uvicorn not available") from e

    async def _stop_server(self) -> None:
        """Stop HTTP server."""
        if self._server:
            logger.info("Stopping HTTP server")
            self._server.should_exit = True
            if hasattr(self, "_server_task"):
                await self._server_task
            logger.info("HTTP server stopped")


async def main() -> None:
    """Main entry point for NPU service."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Intel NPU Inference Service")
    parser.add_argument("--port", type=int, default=int(os.getenv("NPU_SERVICE_PORT", "52416")), help="Port to listen on")
    parser.add_argument("--cache-dir", type=str, default=None, help="Model cache directory")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start service
    service = NPUInferenceService(port=args.port, cache_dir=args.cache_dir)

    try:
        await service.start()

        # Run until interrupted
        logger.info("NPU service running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
