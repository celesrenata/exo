"""Configuration types for inference backend selection.

This module defines the configuration schema for selecting and configuring
different inference backends (MLX, tinygrad, etc.).

Adapted from exo-cuda reference implementation for Intel Arc GPU support.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BackendConfig(BaseModel, frozen=True, strict=True):
    """Configuration for inference backend selection.
    
    This configuration determines which backend to use for inference
    and how to configure it for optimal performance on Intel Arc GPUs.
    
    Attributes:
        backend_name: Name of backend to use ("mlx", "tinygrad", "dummy")
        device: Target device for tinygrad ("GPU", "CPU", "OPENCL", "LEVEL_ZERO")
        runtime: GPU runtime for tinygrad (Intel Arc specific)
        fallback_to_cpu: Whether to fall back to CPU if GPU unavailable
        
    Example:
        >>> # Intel Arc iGPU with Level Zero
        >>> config = BackendConfig(
        ...     backend_name="tinygrad",
        ...     device="GPU",
        ...     runtime="LEVEL_ZERO",
        ...     fallback_to_cpu=True
        ... )
        
        >>> # Intel Arc iGPU with OpenCL fallback
        >>> config = BackendConfig(
        ...     backend_name="tinygrad",
        ...     device="GPU",
        ...     runtime="OPENCL",
        ...     fallback_to_cpu=True
        ... )
    """

    backend_name: Literal["mlx", "tinygrad", "dummy"] = Field(
        default="mlx",
        description="Name of inference backend to use",
    )

    # Tinygrad device configuration for Intel Arc
    device: Literal["GPU", "CPU", "OPENCL", "LEVEL_ZERO"] = Field(
        default="GPU",
        description="Target device for tinygrad execution. GPU auto-detects OpenCL/Level Zero.",
    )

    # Intel Arc GPU runtime selection
    runtime: Literal["LEVEL_ZERO", "OPENCL", "AUTO"] | None = Field(
        default="AUTO",
        description="GPU runtime for Intel Arc. AUTO tries Level Zero first, then OpenCL.",
    )

    # Fallback behavior
    fallback_to_cpu: bool = Field(
        default=True,
        description="Whether to fall back to CPU if GPU is unavailable",
    )
