"""Workload routing logic for NPU inference.

This module determines which tasks should be routed to NPU vs GPU/CPU.
"""

import logging
from typing import Literal

from exo.shared.types.tasks import Task, TextGeneration, ImageGeneration, ImageEdits
from exo.worker.engines.npu.discovery import NPUCapabilities

logger = logging.getLogger(__name__)


WorkloadType = Literal[
    "llm_decode", "embedding", "vision", "audio", "image_generation", "unknown"
]


def classify_workload(task: Task) -> WorkloadType:
    """Classify a task into a workload type.

    Args:
        task: Task to classify

    Returns:
        Workload type classification
    """
    if isinstance(task, TextGeneration):
        # Check if this is an embedding model or LLM
        # Embedding models typically have "embed" in the name
        model_id = str(task.instance_id).lower()

        if "embed" in model_id or "sentence" in model_id:
            return "embedding"
        else:
            return "llm_decode"

    elif isinstance(task, ImageGeneration):
        return "image_generation"

    elif isinstance(task, ImageEdits):
        return "vision"

    else:
        return "unknown"


def should_use_npu(task: Task, npu_caps: NPUCapabilities) -> bool:
    """Determine if a task should be routed to NPU.

    Intel NPU is optimized for:
    - Embedding generation (text embeddings, sentence transformers)
    - Vision models (image classification, object detection)
    - Audio processing (speech recognition, audio classification)
    - Small transformer models (<1B parameters)

    Intel NPU is NOT suitable for:
    - Large LLM decode (use GPU/CPU instead)
    - High-throughput streaming inference
    - Image generation (FLUX, Stable Diffusion)

    Args:
        task: Task to evaluate
        npu_caps: NPU capabilities

    Returns:
        True if task should use NPU, False otherwise
    """
    # NPU must be available
    if not npu_caps.available:
        logger.debug("NPU not available, routing to CPU/GPU")
        return False

    # Classify workload
    workload_type = classify_workload(task)

    # Route based on workload type
    if workload_type == "embedding":
        # Embeddings are ideal for NPU
        logger.debug(f"Routing embedding task to NPU")
        return "embeddings" in npu_caps.supported_model_types

    elif workload_type == "vision":
        # Vision tasks can benefit from NPU
        logger.debug(f"Routing vision task to NPU")
        return "vision" in npu_caps.supported_model_types

    elif workload_type == "audio":
        # Audio processing can benefit from NPU
        logger.debug(f"Routing audio task to NPU")
        return "audio" in npu_caps.supported_model_types

    elif workload_type == "llm_decode":
        # Large LLM decode should stay on GPU/CPU
        # Only route small transformers to NPU
        logger.debug(f"LLM decode task - keeping on GPU/CPU")
        return False

    elif workload_type == "image_generation":
        # Image generation (FLUX, SD) should stay on GPU
        logger.debug(f"Image generation task - keeping on GPU")
        return False

    else:
        # Unknown workload - default to CPU/GPU
        logger.debug(f"Unknown workload type - routing to CPU/GPU")
        return False


def get_fallback_device(task: Task) -> Literal["GPU", "CPU"]:
    """Determine fallback device when NPU is unavailable.

    Args:
        task: Task to evaluate

    Returns:
        Fallback device type
    """
    workload_type = classify_workload(task)

    # Most workloads benefit from GPU if available
    if workload_type in ["llm_decode", "image_generation", "vision"]:
        return "GPU"

    # Embeddings and audio can run efficiently on CPU
    elif workload_type in ["embedding", "audio"]:
        return "CPU"

    else:
        return "GPU"


class NPURouter:
    """Router for NPU workload management.

    This class manages routing decisions and fallback logic for NPU inference.
    """

    def __init__(self, npu_caps: NPUCapabilities, npu_service_available: bool = False):
        """Initialize NPU router.

        Args:
            npu_caps: NPU capabilities
            npu_service_available: Whether NPU service is running and accessible
        """
        self.npu_caps = npu_caps
        self.npu_service_available = npu_service_available

        # Track routing statistics
        self.routed_to_npu = 0
        self.routed_to_fallback = 0
        self.npu_failures = 0

    def route_task(self, task: Task) -> Literal["NPU", "GPU", "CPU"]:
        """Route a task to appropriate device.

        Args:
            task: Task to route

        Returns:
            Target device for task execution
        """
        # Check if NPU service is available
        if not self.npu_service_available:
            logger.debug("NPU service not available, using fallback")
            self.routed_to_fallback += 1
            return get_fallback_device(task)

        # Check if task should use NPU
        if should_use_npu(task, self.npu_caps):
            logger.debug(f"Routing task to NPU")
            self.routed_to_npu += 1
            return "NPU"
        else:
            logger.debug(f"Routing task to fallback device")
            self.routed_to_fallback += 1
            return get_fallback_device(task)

    def mark_npu_failure(self) -> None:
        """Mark an NPU inference failure.

        This can be used to implement circuit breaker logic if NPU
        becomes unreliable.
        """
        self.npu_failures += 1
        logger.warning(f"NPU failure recorded (total: {self.npu_failures})")

        # If too many failures, disable NPU routing temporarily
        if self.npu_failures >= 5:
            logger.error("Too many NPU failures, disabling NPU routing")
            self.npu_service_available = False

    def reset_failure_count(self) -> None:
        """Reset NPU failure count.

        Call this after successful NPU inference to reset circuit breaker.
        """
        if self.npu_failures > 0:
            logger.info("Resetting NPU failure count after successful inference")
            self.npu_failures = 0

    def get_stats(self) -> dict[str, int]:
        """Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        return {
            "routed_to_npu": self.routed_to_npu,
            "routed_to_fallback": self.routed_to_fallback,
            "npu_failures": self.npu_failures,
        }


async def check_npu_service_health(host: str = "localhost", port: int = 52416) -> bool:
    """Check if NPU service is healthy and available.

    Args:
        host: Service host
        port: Service port

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        from exo.worker.engines.npu.protocol import NPUServiceClient

        async with NPUServiceClient(host=host, port=port) as client:
            health = await client.health()
            return health.status == "healthy" and health.npu_available

    except Exception as e:
        logger.debug(f"NPU service health check failed: {e}")
        return False
