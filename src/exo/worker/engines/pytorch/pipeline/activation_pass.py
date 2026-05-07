"""Inter-stage activation transfer for passing hidden states between pipeline stages.

Handles sending and receiving hidden state tensors between adjacent pipeline
stages using the Communicator from the distributed/ package for actual tensor
transfer. Each transfer includes request_id and sequence_position metadata
for tracing and coordination.

Requirements: 5.4, 5.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Communicator protocol — defines the interface activation_pass expects.
# The actual Communicator implementation lives in distributed/communicator.py.
# ---------------------------------------------------------------------------


class CommunicatorProtocol(Protocol):
    """Protocol defining the tensor communication interface.

    The real Communicator class (distributed/communicator.py) must satisfy
    this protocol. Using a Protocol here avoids circular imports and allows
    activation_pass to be tested independently.
    """

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Send a tensor to the destination rank."""
        ...

    def recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        target_device: str,
    ) -> torch.Tensor:
        """Receive a tensor from the source rank and move to target device."""
        ...


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationMessage:
    """Activation tensor metadata passed between pipeline stages.

    This dataclass carries the metadata associated with an activation
    transfer. The actual tensor data is sent separately via the
    Communicator's send_tensor/recv_tensor methods.
    """

    request_id: str
    sequence_position: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str  # "float16", "bfloat16", "float32"


# ---------------------------------------------------------------------------
# Dtype mapping helpers
# ---------------------------------------------------------------------------

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch dtype to its string representation."""
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"Unsupported tensor dtype for activation transfer: {dtype}. "
            f"Supported: {list(_DTYPE_TO_STR.values())}"
        )
    return _DTYPE_TO_STR[dtype]


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string dtype representation to a torch dtype."""
    if dtype_str not in _STR_TO_DTYPE:
        raise ValueError(
            f"Unknown dtype string: {dtype_str!r}. "
            f"Supported: {list(_STR_TO_DTYPE.keys())}"
        )
    return _STR_TO_DTYPE[dtype_str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_activation(
    communicator: CommunicatorProtocol,
    tensor: torch.Tensor,
    dst_rank: int,
    request_id: str,
    sequence_position: int,
) -> ActivationMessage:
    """Send a hidden state tensor to the next pipeline stage.

    Sends the activation tensor to ``dst_rank`` via the communicator and
    returns an ActivationMessage describing the transfer metadata.

    Args:
        communicator: The distributed communicator for tensor transfer.
        tensor: The hidden state tensor to send. Must be float16, bfloat16,
            or float32.
        dst_rank: The rank of the destination pipeline stage.
        request_id: Unique identifier for the inference request.
        sequence_position: The token position in the sequence being processed.

    Returns:
        An ActivationMessage containing the metadata of the sent activation.

    Raises:
        ValueError: If the tensor dtype is not supported.
    """
    dtype_str = _dtype_to_str(tensor.dtype)
    shape = tuple(tensor.shape)

    message = ActivationMessage(
        request_id=request_id,
        sequence_position=sequence_position,
        tensor_shape=shape,
        tensor_dtype=dtype_str,
    )

    logger.debug(
        "Sending activation to rank %d: request_id=%s, seq_pos=%d, "
        "shape=%s, dtype=%s",
        dst_rank,
        request_id,
        sequence_position,
        shape,
        dtype_str,
    )

    communicator.send_tensor(tensor, dst_rank)

    logger.debug(
        "Activation sent to rank %d: request_id=%s, seq_pos=%d",
        dst_rank,
        request_id,
        sequence_position,
    )

    return message


def recv_activation(
    communicator: CommunicatorProtocol,
    src_rank: int,
    shape: tuple[int, ...],
    dtype: str,
    target_device: str,
    request_id: str,
    sequence_position: int,
) -> tuple[torch.Tensor, ActivationMessage]:
    """Receive a hidden state tensor from the previous pipeline stage.

    Receives an activation tensor from ``src_rank`` via the communicator
    and moves it to the specified target device.

    Args:
        communicator: The distributed communicator for tensor transfer.
        src_rank: The rank of the source pipeline stage.
        shape: Expected shape of the incoming tensor.
        dtype: Expected dtype as a string ("float16", "bfloat16", "float32").
        target_device: Device to place the received tensor on (e.g., "xpu:0").
        request_id: Unique identifier for the inference request.
        sequence_position: The token position in the sequence being processed.

    Returns:
        A tuple of (received_tensor, ActivationMessage) where the tensor is
        on the target_device and the message contains transfer metadata.

    Raises:
        ValueError: If the dtype string is not recognized.
    """
    torch_dtype = _str_to_dtype(dtype)

    message = ActivationMessage(
        request_id=request_id,
        sequence_position=sequence_position,
        tensor_shape=shape,
        tensor_dtype=dtype,
    )

    logger.debug(
        "Receiving activation from rank %d: request_id=%s, seq_pos=%d, "
        "shape=%s, dtype=%s, target_device=%s",
        src_rank,
        request_id,
        sequence_position,
        shape,
        dtype,
        target_device,
    )

    tensor = communicator.recv_tensor(shape, torch_dtype, src_rank, target_device)

    logger.debug(
        "Activation received from rank %d: request_id=%s, seq_pos=%d, "
        "actual_shape=%s, device=%s",
        src_rank,
        request_id,
        sequence_position,
        tuple(tensor.shape),
        tensor.device,
    )

    return tensor, message
