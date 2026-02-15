"""
PyTorch + IPEX backend for Intel Arc GPU support.

This module provides Intel Arc GPU support for exo using PyTorch and
Intel Extension for PyTorch (IPEX).
"""

from .device_manager import DeviceInfo, DeviceManager, DeviceType
from .model_loader import ModelLoader, TransformerShard

__all__ = [
    "DeviceManager",
    "DeviceInfo",
    "DeviceType",
    "ModelLoader",
    "TransformerShard",
]
