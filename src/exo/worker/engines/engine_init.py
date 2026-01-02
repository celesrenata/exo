"""Engine initialization that supports multiple backends."""

from typing import Any, Callable

from exo.shared.types.worker.instances import BoundInstance


def initialize_engine(
    bound_instance: BoundInstance, connect_only: bool = False
) -> tuple[Any, Any, Callable] | Any:
    """
    Initialize the appropriate inference engine based on system capabilities.

    Args:
        bound_instance: The bound instance configuration
        connect_only: If True, only establish connection (for MLX group setup)

    Returns:
        If connect_only=True: group object (for MLX) or None
        If connect_only=False: tuple of (model, tokenizer, sampler)
    """
    # Import here to avoid circular imports
    from exo.worker.engines.engine_utils import select_best_engine
    from exo.worker.runner.bootstrap import logger

    engine_type = select_best_engine()

    if connect_only:
        logger.info(f"Connecting to {engine_type} engine")
        if engine_type == "mlx":
            from exo.worker.engines.mlx.utils_mlx import initialize_mlx

            return initialize_mlx(bound_instance)
        else:
            # CPU engines don't need separate connection step
            return None

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
    # Import here to avoid circular imports
    from exo.worker.engines.engine_utils import select_best_engine
    from exo.worker.runner.bootstrap import logger

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
    # Import here to avoid circular imports
    from exo.worker.engines.engine_utils import select_best_engine
    from exo.worker.runner.bootstrap import logger

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
