"""Utilities for detecting and selecting inference engines."""

import os
from typing import Literal

from exo.worker.runner.bootstrap import logger

EngineType = Literal["mlx", "torch", "cpu"]


def detect_available_engines() -> list[EngineType]:
    """Detect which inference engines are available on this system."""
    available = []

    # Check for MLX (Apple Silicon)
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            available.append("mlx")
            logger.info("MLX engine available (Apple Silicon)")
        else:
            logger.info("MLX installed but Metal not available")
    except ImportError:
        logger.info("MLX not available")

    # Check for PyTorch
    try:
        import torch

        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x + 1
        available.append("torch")
        logger.info("PyTorch engine available")
    except ImportError:
        logger.info("PyTorch not available")

    # CPU fallback is always available if torch is available
    if "torch" in available:
        available.append("cpu")

    return available


def select_best_engine() -> EngineType:
    """Select the best available inference engine."""
    available = detect_available_engines()

    # Force engine selection via environment variable
    forced_engine = os.getenv("EXO_ENGINE")
    if forced_engine:
        if forced_engine in available:
            logger.info(f"Using forced engine: {forced_engine}")
            return forced_engine
        else:
            logger.warning(
                f"Forced engine {forced_engine} not available, falling back to auto-selection"
            )

    # Preference order: MLX > PyTorch > CPU
    if "mlx" in available:
        logger.info("Selected MLX engine (best performance on Apple Silicon)")
        return "mlx"
    elif "torch" in available:
        logger.info("Selected PyTorch engine (CPU inference)")
        return "torch"
    elif "cpu" in available:
        logger.info("Selected CPU engine (PyTorch CPU fallback)")
        return "cpu"
    else:
        raise RuntimeError(
            "No inference engines available. Please install MLX (Apple Silicon) or PyTorch."
        )


def is_model_compatible(model_id: str, engine_type: EngineType) -> bool:
    """Check if a model is compatible with a specific engine."""
    model_id_lower = model_id.lower()

    if engine_type == "mlx":
        # MLX models are from mlx-community
        return "mlx-community/" in model_id_lower
    elif engine_type in ["torch", "cpu"]:
        # CPU/PyTorch models are standard HuggingFace models (not mlx-community)
        return "mlx-community/" not in model_id_lower

    return False


def get_compatible_models(available_models: list[str]) -> list[str]:
    """Filter models based on available engines."""
    selected_engine = select_best_engine()
    compatible = []

    for model_id in available_models:
        if is_model_compatible(model_id, selected_engine):
            compatible.append(model_id)

    logger.info(
        f"Found {len(compatible)} models compatible with {selected_engine} engine"
    )
    return compatible


def get_engine_info() -> dict[str, any]:
    """Get information about available engines."""
    info = {
        "available_engines": detect_available_engines(),
        "selected_engine": None,
        "mlx_available": False,
        "torch_available": False,
        "cpu_available": False,
    }

    try:
        selected = select_best_engine()
        info["selected_engine"] = selected
    except RuntimeError:
        pass

    # MLX info
    try:
        import mlx.core as mx

        info["mlx_available"] = mx.metal.is_available()
        if info["mlx_available"]:
            info["mlx_device_info"] = mx.metal.device_info()
    except ImportError:
        pass

    # PyTorch info
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cpu_count"] = torch.get_num_threads()
    except ImportError:
        pass

    info["cpu_available"] = info["torch_available"]

    return info
