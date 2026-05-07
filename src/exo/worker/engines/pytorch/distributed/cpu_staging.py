"""CPU-staged tensor transfer for Gloo backend communication.

Provides functions to move tensors through CPU memory as an intermediate
step when transferring between CUDA and XPU nodes via the Gloo backend.
On Intel iGPU (shared memory architecture), CPU↔GPU copy is nearly free
since they share the same physical memory.

Requirements: 6.7
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def stage_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move a GPU tensor to CPU for Gloo send.

    If the tensor is already on CPU, returns it as-is (no-op).
    Ensures the returned tensor is contiguous for efficient transfer.

    Args:
        tensor: The tensor to stage to CPU. May be on any device.

    Returns:
        A contiguous CPU tensor with the same dtype and shape.
    """
    if tensor.device.type == "cpu":
        logger.debug(
            "stage_to_cpu: tensor already on CPU, shape=%s dtype=%s",
            tensor.shape,
            tensor.dtype,
        )
        return tensor.contiguous()

    logger.debug(
        "stage_to_cpu: moving tensor from %s to CPU, shape=%s dtype=%s",
        tensor.device,
        tensor.shape,
        tensor.dtype,
    )
    return tensor.detach().cpu().contiguous()


def unstage_from_cpu(tensor: torch.Tensor, target_device: str) -> torch.Tensor:
    """Move a CPU tensor back to the target GPU device after Gloo recv.

    If the tensor is already on the target device, returns it as-is (no-op).
    Handles mixed CUDA↔XPU transfers through the CPU intermediate — the
    target_device can be any valid torch device string (e.g. "xpu:0", "cuda:0").

    Args:
        tensor: The CPU tensor received via Gloo.
        target_device: Device string to move the tensor to (e.g. "xpu:0", "cuda:0").

    Returns:
        A tensor on the target device with the same dtype and shape.
    """
    target = torch.device(target_device)

    if tensor.device == target:
        logger.debug(
            "unstage_from_cpu: tensor already on %s, shape=%s dtype=%s",
            target_device,
            tensor.shape,
            tensor.dtype,
        )
        return tensor

    logger.debug(
        "unstage_from_cpu: moving tensor from %s to %s, shape=%s dtype=%s",
        tensor.device,
        target_device,
        tensor.shape,
        tensor.dtype,
    )
    return tensor.to(target_device)
