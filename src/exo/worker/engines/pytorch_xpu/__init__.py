"""
PyTorch XPU Inference Backend

This package provides Intel Arc GPU support for exo using native PyTorch XPU
(2.11+). No Intel Extension for PyTorch (IPEX) dependency — all Intel GPU
operations use native torch.xpu APIs.

Components:
- PyTorchXPUBackend: Main inference engine
- DeviceManager: Device detection and selection
- ModelLoader: Model loading and optimization
- KVCacheManager: KV cache management
- TokenGenerator: Token sampling with temperature, top-k, top-p
"""

from exo.worker.engines.pytorch_xpu.device_manager import DeviceManager, DeviceType
from exo.worker.engines.pytorch_xpu.errors import (
    CacheError,
    DeviceError,
    InferenceError,
    ModelError,
)
from exo.worker.engines.pytorch_xpu.kv_cache_manager import KVCacheManager
from exo.worker.engines.pytorch_xpu.model_loader import ModelLoader
from exo.worker.engines.pytorch_xpu.pytorch_xpu_backend import PyTorchXPUBackend
from exo.worker.engines.pytorch_xpu.token_generator import (
    SamplingResult,
    TokenGenerator,
)

__all__ = [
    "PyTorchXPUBackend",
    "DeviceManager",
    "DeviceType",
    "ModelLoader",
    "KVCacheManager",
    "TokenGenerator",
    "SamplingResult",
    "DeviceError",
    "ModelError",
    "InferenceError",
    "CacheError",
]
