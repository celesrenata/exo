"""Engine initialization that supports multiple backends."""

from typing import Any, Callable, Union

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.engines.engine_utils import select_best_engine
from exo.worker.runner.bootstrap import logger


def initialize_engine(bound_instance: BoundInstance) -> tuple[Any, Any, Callable]:
    """
    Initialize the appropriate inference engine based on system capabilities.
    
    Returns:
        tuple: (model, tokenizer, sampler) - types depend on the selected engine
    """
    engine_type = select_best_engine()
    
    logger.info(f"Initializing {engine_type} engine")
    
    if engine_type == "mlx":
        from exo.worker.engines.mlx.utils_mlx import initialize_mlx
        return initialize_mlx(bound_instance)
    
    elif engine_type in ["torch", "cpu"]:
        from exo.worker.engines.torch.utils_torch import initialize_torch
        return initialize_torch(bound_instance)
    
    else:
        raise RuntimeError(f"Unsupported engine type: {engine_type}")


def warmup_engine(model: Any, tokenizer: Any, sampler: Callable) -> int:
    """
    Warm up the selected inference engine.
    
    Returns:
        int: Number of tokens generated during warmup
    """
    engine_type = select_best_engine()
    
    logger.info(f"Warming up {engine_type} engine")
    
    if engine_type == "mlx":
        from exo.worker.engines.mlx.generator.generate import warmup_inference
        return warmup_inference(model, tokenizer, sampler)
    
    elif engine_type in ["torch", "cpu"]:
        from exo.worker.engines.torch.generator.generate import warmup_inference
        return warmup_inference(model, tokenizer, sampler)
    
    else:
        raise RuntimeError(f"Unsupported engine type: {engine_type}")


def generate_with_engine(
    model: Any,
    tokenizer: Any,
    sampler: Callable,
    task: Any,
):
    """
    Generate text using the selected inference engine.
    
    Returns:
        Generator yielding GenerationResponse objects
    """
    engine_type = select_best_engine()
    
    logger.info(f"Generating with {engine_type} engine")
    
    if engine_type == "mlx":
        from exo.worker.engines.mlx.generator.generate import mlx_generate
        return mlx_generate(model, tokenizer, sampler, task)
    
    elif engine_type in ["torch", "cpu"]:
        from exo.worker.engines.torch.generator.generate import torch_generate
        return torch_generate(model, tokenizer, sampler, task)
    
    else:
        raise RuntimeError(f"Unsupported engine type: {engine_type}")