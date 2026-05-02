"""
PyTorch XPU Inference Backend

This package provides Intel Arc GPU support for exo using native PyTorch XPU
(2.11+). No Intel Extension for PyTorch (IPEX) dependency — all Intel GPU
operations use native torch.xpu APIs.

Components:
- PyTorchIPEXBackend: Main inference engine
- DeviceManager: Device detection and selection
- ModelLoader: Model loading and optimization
- KVCacheManager: KV cache management
- TokenGenerator: Token sampling with temperature, top-k, top-p
"""

from exo.worker.engines.pytorch_ipex.device_manager import DeviceManager, DeviceType
from exo.worker.engines.pytorch_ipex.errors import (
    CacheError,
    DeviceError,
    InferenceError,
    ModelError,
)
from exo.worker.engines.pytorch_ipex.kv_cache_manager import KVCacheManager
from exo.worker.engines.pytorch_ipex.model_loader import ModelLoader
from exo.worker.engines.pytorch_ipex.pytorch_ipex_backend import PyTorchIPEXBackend
from exo.worker.engines.pytorch_ipex.token_generator import (
    SamplingResult,
    TokenGenerator,
)

__all__ = [
    "PyTorchIPEXBackend",
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
