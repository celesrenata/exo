"""
PyTorch XPU Inference Backend

This package provides Intel Arc GPU support for exo using native PyTorch XPU
(2.11+). No Intel Extension for PyTorch (IPEX) dependency — all Intel GPU
operations use native torch.xpu APIs.

Components:
- PyTorchXPUEngine: Engine implementation for the upstream runner
- PyTorchXPUBuilder: Builder implementation for the upstream runner
- DeviceManager: Device detection and selection
- ModelLoader: Model loading and optimization
- KVCacheManager: KV cache management
- TokenGenerator: Token sampling with temperature, top-k, top-p
"""

from exo.worker.engines.pytorch_xpu.errors import (
    CacheError,
    DeviceError,
    InferenceError,
    ModelError,
)

__all__ = [
    "DeviceError",
    "ModelError",
    "InferenceError",
    "CacheError",
]
