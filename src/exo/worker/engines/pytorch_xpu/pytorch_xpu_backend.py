"""
PyTorch XPU Inference Backend

This module provides the main inference engine for Intel Arc GPUs using
native PyTorch XPU (2.11+). No IPEX dependency.

Requirements addressed:
- 3.1: InferenceEngine protocol implementation
- 3.2: Async inference execution
- 3.3: Token sampling
- 3.4: Model lifecycle management
- 3.5: Error handling
"""

import time
from typing import Any, Optional, final

import numpy as np
from loguru import logger

from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.engines.base import InferenceBackend
from exo.worker.engines.pytorch_xpu.device_manager import DeviceManager
from exo.worker.engines.pytorch_xpu.errors import (
    CacheError,
    DeviceError,
    InferenceError,
    ModelError,
)
from exo.worker.engines.pytorch_xpu.health_check import HealthChecker
from exo.worker.engines.pytorch_xpu.kv_cache_manager import KVCacheManager
from exo.worker.engines.pytorch_xpu.logging_config import (
    LogContext,
    log_device_info,
    log_inference_metrics,
    log_model_info,
)
from exo.worker.engines.pytorch_xpu.model_loader import ModelLoader
from exo.worker.engines.pytorch_xpu.performance_metrics import (
    PerformanceMetricsCollector,
    collect_gpu_metrics,
)


@final
class PyTorchXPUBackend(InferenceBackend):
    """
    PyTorch XPU inference backend for Intel Arc GPUs.

    This class implements the InferenceBackend protocol and provides:
    - Automatic device detection and selection (Intel Arc > NVIDIA > CPU)
    - Model loading with native PyTorch XPU device placement
    - Async inference execution with KV cache
    - Token sampling with temperature and top-p
    - Graceful error handling and fallbacks

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """

    def __init__(self, shard_downloader: Optional[Any] = None) -> None:
        """
        Initialize the PyTorchXPUBackend.

        Sets up:
        - Device manager for GPU detection and selection
        - Model loader for HuggingFace model loading
        - KV cache manager for efficient inference
        - Performance metrics collector
        - Health checker
        - Logging configuration

        Args:
            shard_downloader: Optional shard downloader for model weights (not used yet)

        Requirements: 3.1, 9.1, 9.2, 10.5
        """
        logger.info("Initializing PyTorchXPUBackend")

        # Store shard downloader for future use
        self._shard_downloader = shard_downloader

        # Initialize components
        self._device_manager = DeviceManager()
        self._model_loader = ModelLoader()

        # Select device
        device_type, device_id = self._device_manager.select_device()
        self._device_type = device_type
        self._device_id = device_id

        # Log device information
        total_memory, free_memory = self._device_manager.get_device_memory(
            device_type, device_id
        )
        log_device_info(
            device_type=device_type,
            device_id=device_id,
            device_name=f"{device_type}:{device_id}",
            total_memory_gb=total_memory / (1024**3),
            free_memory_gb=free_memory / (1024**3),
        )

        # Initialize KV cache manager
        self._cache_manager = KVCacheManager(
            device_type=device_type,
            device_id=device_id,
            memory_threshold_percent=80.0,
        )

        # Initialize performance metrics collector
        self._metrics_collector = PerformanceMetricsCollector(
            device_type=device_type,
            device_id=device_id,
        )

        # Initialize health checker
        self._health_checker = HealthChecker(
            device_type=device_type,
            device_id=device_id,
        )

        # Cache for loaded models: shard_key -> (model, tokenizer)
        self._loaded_models: dict[str, tuple[Any, Any]] = {}

        # Current shard metadata
        self._current_shard: Optional[ShardMetadata] = None

        # Try to import torch
        self._torch_available = False
        try:
            import torch  # type: ignore

            self._torch_available = True
            self._torch = torch
            logger.info(f"PyTorch available: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch not available - backend cannot function")
            raise RuntimeError("PyTorch is required for PyTorchXPUBackend")

        logger.info("PyTorchXPUBackend initialized successfully")

    def _get_shard_key(self, shard_metadata: ShardMetadata) -> str:
        """
        Generate a unique key for a shard.

        Args:
            shard_metadata: Metadata describing the model shard

        Returns:
            Unique string key for the shard
        """
        return (
            f"{shard_metadata.model_card.model_id}_"
            f"{shard_metadata.start_layer}_{shard_metadata.end_layer}"
        )

    async def encode(self, shard_metadata: ShardMetadata, prompt: str) -> np.ndarray:
        """
        Encode a text prompt into tokens.

        Args:
            shard_metadata: Metadata describing the model shard
            prompt: Text prompt to encode

        Returns:
            Token IDs as numpy array

        Raises:
            ModelError: If encoding fails

        Requirements: 3.1
        """
        try:
            model_id = str(shard_metadata.model_card.model_id)
            tokens = await self._model_loader.encode(model_id, prompt)
            logger.debug(f"Encoded prompt to {len(tokens)} tokens")
            return tokens

        except Exception as e:
            logger.error(f"Failed to encode prompt: {e}")
            raise ModelError(
                message="Failed to encode prompt",
                model_id=str(shard_metadata.model_card.model_id),
                original_error=e,
            ) from e

    async def decode(self, shard_metadata: ShardMetadata, tokens: np.ndarray) -> str:
        """
        Decode tokens back into text.

        Args:
            shard_metadata: Metadata describing the model shard
            tokens: Token IDs to decode

        Returns:
            Decoded text string

        Raises:
            ModelError: If decoding fails

        Requirements: 3.1
        """
        try:
            model_id = str(shard_metadata.model_card.model_id)
            text = await self._model_loader.decode(model_id, tokens)
            logger.debug(f"Decoded {len(tokens)} tokens to text")
            return text

        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            raise ModelError(
                message="Failed to decode tokens",
                model_id=str(shard_metadata.model_card.model_id),
                original_error=e,
            ) from e


    async def load_checkpoint(self, shard_metadata: ShardMetadata, path: str) -> None:
        """
        Load model weights from checkpoint.

        This method ensures the shard is loaded. The actual checkpoint loading
        is handled by the model loader from HuggingFace.

        Args:
            shard_metadata: Metadata describing the model shard
            path: Path to checkpoint file (unused, models loaded from HuggingFace)

        Requirements: 3.1
        """
        logger.info(f"Loading checkpoint for shard: {shard_metadata.model_card.model_id}")
        await self._ensure_shard(shard_metadata)
        logger.info("Checkpoint loaded successfully")

    async def _ensure_shard(self, shard_metadata: ShardMetadata) -> tuple[Any, Any]:
        """
        Ensure the correct shard is loaded.

        This method:
        1. Checks if the shard is already loaded
        2. Downloads and loads the model if needed
        3. Applies native PyTorch XPU device placement
        4. Caches the model instance

        Args:
            shard_metadata: Metadata describing the model shard

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelError: If model loading fails
            DeviceError: If device is unavailable

        Requirements: 2.1, 2.2, 9.1, 9.2
        """
        shard_key = self._get_shard_key(shard_metadata)

        # Check if already loaded
        if shard_key in self._loaded_models:
            logger.debug(f"Shard already loaded: {shard_key}")
            self._current_shard = shard_metadata
            return self._loaded_models[shard_key]

        logger.info(f"Loading shard: {shard_key}")
        load_start_time = time.time()

        try:
            # Verify device is still available
            if not self._device_manager.is_device_available(
                self._device_type, self._device_id
            ):
                raise DeviceError(
                    message="Device is no longer available",
                    device_type=self._device_type,
                    device_id=self._device_id,
                )

            # Load model and tokenizer
            model, tokenizer = await self._model_loader.load_model(
                shard_metadata=shard_metadata,
                device_type=self._device_type,
                device_id=self._device_id,
            )

            # Cache the loaded model
            self._loaded_models[shard_key] = (model, tokenizer)
            self._current_shard = shard_metadata

            # Record model loaded in health checker
            self._health_checker.record_model_loaded()

            # Log model loading metrics
            load_time = time.time() - load_start_time
            log_model_info(
                model_id=str(shard_metadata.model_card.model_id),
                shard_info=f"layers {shard_metadata.start_layer}-{shard_metadata.end_layer}",
                device=f"{self._device_type}:{self._device_id}",
                optimization_applied=True,
                load_time_seconds=load_time,
            )

            logger.info(f"Successfully loaded and cached shard: {shard_key}")

            return model, tokenizer

        except DeviceError:
            # Re-raise device errors
            self._health_checker.record_inference_error("Device error during model loading")
            raise
        except Exception as e:
            logger.error(f"Failed to load shard {shard_key}: {e}")
            self._health_checker.record_inference_error(f"Model loading failed: {e}")
            raise ModelError(
                message="Failed to load shard",
                model_id=str(shard_metadata.model_card.model_id),
                shard_info=f"layers {shard_metadata.start_layer}-{shard_metadata.end_layer}",
                original_error=e,
            ) from e

    async def infer_tensor(
        self,
        request_id: str,
        shard_metadata: ShardMetadata,
        input_data: np.ndarray,
        inference_state: Optional[dict] = None,
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Execute tensor inference.

        This method:
        1. Ensures the correct shard is loaded
        2. Gets or creates KV cache for the request
        3. Executes forward pass with cache
        4. Returns output and updated state
        5. Records performance metrics

        Args:
            request_id: Unique identifier for this inference request
            shard_metadata: Metadata describing the model shard
            input_data: Input tensor as numpy array (shape: [batch_size, seq_len])
            inference_state: Optional state from previous inference (KV cache, etc.)

        Returns:
            Tuple of (output_data, new_inference_state)

        Raises:
            InferenceError: If inference fails
            DeviceError: If device error occurs
            ModelError: If model error occurs

        Requirements: 3.2, 4.1, 9.1, 9.2
        """
        inference_start_time = time.time()
        
        # Use log context for this inference
        with LogContext(request_id=request_id, model_id=str(shard_metadata.model_card.model_id)):
            try:
                logger.debug(
                    f"Starting inference for request {request_id}, "
                    f"input shape: {input_data.shape}"
                )

                # Ensure correct shard is loaded
                model, tokenizer = await self._ensure_shard(shard_metadata)

                # Convert input to torch tensor
                device = self._torch.device(f"{self._device_type}:{self._device_id}")
                
                try:
                    input_tensor = self._torch.from_numpy(input_data).long().to(device)
                except Exception as e:
                    raise DeviceError(
                        message="Failed to move input tensor to device",
                        device_type=self._device_type,
                        device_id=self._device_id,
                        original_error=e,
                    ) from e

                # Get or create KV cache
                cache = self._cache_manager.get_cache(request_id)
                past_key_values = None
                cache_hit = cache is not None

                try:
                    if cache is None:
                        # Create new cache
                        logger.debug(f"Creating new cache for request {request_id}")
                        cache = self._cache_manager.create_cache(
                            request_id=request_id,
                            num_layers=shard_metadata.n_layers,
                            max_length=8192,
                        )
                    else:
                        # Extract past key values from cache
                        if cache.keys[0] is not None:
                            past_key_values = list(zip(cache.keys, cache.values, strict=False))
                            logger.debug(
                                f"Using cached KV for request {request_id}, "
                                f"position: {cache.position}"
                            )
                except Exception as e:
                    raise CacheError(
                        message="Failed to get or create KV cache",
                        request_id=request_id,
                        original_error=e,
                    ) from e

                # Execute forward pass
                try:
                    with self._torch.no_grad():
                        if hasattr(model, "forward"):
                            # TransformerShard or custom model
                            output, new_past_key_values = model.forward(
                                input_data=input_tensor,
                                past_key_values=past_key_values,
                            )
                        else:
                            # Standard HuggingFace model
                            outputs = model(
                                input_ids=input_tensor,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            output = outputs.logits
                            new_past_key_values = outputs.past_key_values
                except Exception as e:
                    raise InferenceError(
                        message="Forward pass failed",
                        request_id=request_id,
                        model_id=str(shard_metadata.model_card.model_id),
                        original_error=e,
                    ) from e

                # Check for NaN or invalid outputs
                if self._torch.isnan(output).any():
                    raise InferenceError(
                        message="Model output contains NaN values",
                        request_id=request_id,
                        model_id=str(shard_metadata.model_card.model_id),
                    )

                # Update KV cache
                try:
                    if new_past_key_values is not None:
                        for layer_idx, (new_key, new_value) in enumerate(new_past_key_values):
                            self._cache_manager.update_cache(
                                request_id=request_id,
                                layer_idx=layer_idx,
                                new_key=new_key,
                                new_value=new_value,
                            )
                except Exception as e:
                    raise CacheError(
                        message="Failed to update KV cache",
                        request_id=request_id,
                        original_error=e,
                    ) from e

                # Convert output to numpy
                try:
                    output_np = output.cpu().numpy()
                except Exception as e:
                    raise InferenceError(
                        message="Failed to convert output to numpy",
                        request_id=request_id,
                        model_id=str(shard_metadata.model_card.model_id),
                        original_error=e,
                    ) from e

                # Create new inference state
                new_state: dict[str, object] = {
                    "request_id": request_id,
                    "position": cache.position + input_data.shape[1],
                }

                # Record performance metrics
                inference_duration = time.time() - inference_start_time
                tokens_generated = output_np.shape[1] if len(output_np.shape) > 1 else 1
                
                # Collect GPU metrics
                gpu_metrics = collect_gpu_metrics(self._device_type, self._device_id)
                memory_used_mb = gpu_metrics.get("memory_used_mb", 0.0)
                gpu_utilization = gpu_metrics.get("gpu_utilization_percent")
                
                # Record metrics
                self._metrics_collector.record_inference(
                    request_id=request_id,
                    tokens=tokens_generated,
                    duration=inference_duration,
                    memory_used=memory_used_mb,
                    gpu_utilization=gpu_utilization,
                    cache_hit=cache_hit,
                )
                
                # Log inference metrics
                log_inference_metrics(
                    request_id=request_id,
                    tokens_generated=tokens_generated,
                    duration_seconds=inference_duration,
                    tokens_per_second=tokens_generated / inference_duration if inference_duration > 0 else 0.0,
                    memory_used_mb=memory_used_mb,
                    cache_hit=cache_hit,
                )
                
                # Record success in health checker
                self._health_checker.record_inference_success()

                logger.debug(
                    f"Inference complete for request {request_id}, "
                    f"output shape: {output_np.shape}"
                )

                return output_np, new_state

            except (DeviceError, ModelError, InferenceError, CacheError) as e:
                # Re-raise our custom errors
                logger.error(f"Inference failed for request {request_id}: {e}")
                
                # Record error in health checker
                self._health_checker.record_inference_error(str(e))
                
                # Clean up cache on error
                try:
                    self._cache_manager.evict_cache(request_id)
                except Exception:
                    pass  # Ignore cleanup errors
                raise
            except Exception as e:
                # Catch any unexpected errors
                logger.error(f"Unexpected error during inference for request {request_id}: {e}")
                
                # Record error in health checker
                self._health_checker.record_inference_error(f"Unexpected error: {e}")
                
                # Clean up cache on error
                try:
                    self._cache_manager.evict_cache(request_id)
                except Exception:
                    pass  # Ignore cleanup errors
                raise InferenceError(
                    message="Unexpected inference error",
                    request_id=request_id,
                    model_id=str(shard_metadata.model_card.model_id),
                    original_error=e,
                ) from e

    async def sample(self, logits: np.ndarray) -> np.ndarray:
        """
        Sample tokens from logits.

        This method applies temperature scaling and top-p (nucleus) sampling
        to generate the next token.

        Args:
            logits: Logit values from model output (shape: [batch_size, seq_len, vocab_size])

        Returns:
            Sampled token IDs (shape: [batch_size, 1])

        Raises:
            InferenceError: If sampling fails

        Requirements: 3.3
        """
        try:
            # Default sampling parameters
            temperature = 1.0
            top_p = 0.9

            # Convert to torch tensor
            device = self._torch.device(f"{self._device_type}:{self._device_id}")
            
            try:
                logits_tensor = self._torch.from_numpy(logits).to(device)
            except Exception as e:
                raise DeviceError(
                    message="Failed to move logits to device",
                    device_type=self._device_type,
                    device_id=self._device_id,
                    original_error=e,
                ) from e

            # Get logits for last position
            if len(logits_tensor.shape) == 3:
                logits_tensor = logits_tensor[:, -1, :]  # [batch_size, vocab_size]

            # Check for NaN or invalid logits
            if self._torch.isnan(logits_tensor).any():
                raise InferenceError(
                    message="Logits contain NaN values",
                )

            # Apply temperature
            if temperature != 1.0:
                logits_tensor = logits_tensor / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = self._torch.sort(
                    logits_tensor, descending=True
                )
                cumulative_probs = self._torch.cumsum(
                    self._torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_tensor[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = self._torch.nn.functional.softmax(logits_tensor, dim=-1)
            
            # Check for valid probability distribution
            if self._torch.isnan(probs).any() or (probs < 0).any():
                raise InferenceError(
                    message="Invalid probability distribution after softmax",
                )
            
            sampled_token = self._torch.multinomial(probs, num_samples=1)

            # Convert to numpy
            try:
                sampled_token_np = sampled_token.cpu().numpy()
            except Exception as e:
                raise InferenceError(
                    message="Failed to convert sampled token to numpy",
                    original_error=e,
                ) from e

            logger.debug(f"Sampled token: {sampled_token_np}")

            return sampled_token_np

        except (DeviceError, InferenceError):
            # Re-raise our custom errors
            raise
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise InferenceError(
                message="Unexpected sampling error",
                original_error=e,
            ) from e

    def get_stats(self) -> dict[str, object]:
        """
        Get backend statistics.

        Returns:
            Dictionary containing:
            - device_info: Device information
            - cache_stats: KV cache statistics
            - loaded_models: Number of loaded models
            - performance_metrics: Performance statistics
            - health_status: Health check results

        Requirements: 9.2, 10.5
        """
        stats: dict[str, object] = {
            "device_type": self._device_type,
            "device_id": self._device_id,
            "loaded_models": len(self._loaded_models),
        }

        # Add device stats
        device_stats = self._device_manager.get_device_stats(
            self._device_type, self._device_id
        )
        stats["device"] = device_stats

        # Add cache stats
        cache_stats = self._cache_manager.get_stats()
        stats["cache"] = cache_stats

        # Add performance metrics
        performance_stats = self._metrics_collector.get_stats()
        stats["performance"] = performance_stats

        # Add health status
        health_summary = self._health_checker.get_health_summary()
        stats["health"] = health_summary

        return stats

    def get_health_status(self) -> dict[str, Any]:
        """
        Get detailed health status.

        Returns:
            Dictionary with health check results

        Requirements: 10.5
        """
        return self._health_checker.get_health_summary()

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with performance statistics

        Requirements: 9.2
        """
        return self._metrics_collector.get_stats()

    def cleanup(self) -> None:
        """
        Clean up resources.

        This method:
        - Clears all KV caches
        - Unloads all models
        - Frees GPU memory
        - Records cleanup in health checker

        Requirements: 3.4, 10.5
        """
        logger.info("Cleaning up PyTorchXPUBackend")

        # Clear caches
        self._cache_manager.clear_all_caches()

        # Clear loaded models
        self._loaded_models.clear()
        self._current_shard = None

        # Record model unloaded
        self._health_checker.record_model_unloaded()

        # Clear model loader cache
        self._model_loader.clear_cache()

        # Force garbage collection if torch available
        if self._torch_available:
            if self._device_type == "xpu" and hasattr(self._torch, "xpu"):
                self._torch.xpu.empty_cache()  # type: ignore
            elif self._device_type == "cuda":
                self._torch.cuda.empty_cache()

        logger.info("Cleanup complete")
