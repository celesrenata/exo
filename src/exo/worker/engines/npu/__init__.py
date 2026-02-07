"""Intel NPU (Neural Processing Unit) support for exo.

This module provides discovery and integration for Intel NPU hardware
found in Core Ultra processors (also known as iVPU).
"""

from exo.worker.engines.npu.discovery import NPUCapabilities, discover_npu

__all__ = ["NPUCapabilities", "discover_npu"]
