"""Process group management for PyTorch distributed communication.

Manages the PyTorch distributed process group and provides send/recv
operations for passing activation tensors between pipeline stages.
For Gloo backend, tensors are staged through CPU memory. For NCCL,
tensors remain on GPU.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8, 6.9
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, final

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommConfig:
    """Configuration for distributed communication.

    Attributes:
        rank: This node's rank in the process group.
        world_size: Total number of participating nodes.
        master_addr: IP address of the rendezvous master node.
        master_port: Port for the rendezvous endpoint.
        backend: Distributed backend — "gloo" for XPU/mixed, "nccl" for all-CUDA.
        timeout_seconds: Timeout for send/recv operations (default 30s).
        transport: Network transport type for interface binding.
    """

    rank: int
    world_size: int
    master_addr: str
    master_port: int
    backend: Literal["gloo", "nccl"]
    timeout_seconds: int = 30
    transport: Literal["rdma", "ethernet", "lacp"] = "ethernet"


@final
class Communicator:
    """Manages distributed process group and tensor communication.

    Handles initialization of the PyTorch distributed process group,
    point-to-point tensor send/recv operations, and proper cleanup.
    For Gloo backend, GPU tensors are staged through CPU for transfer.
    """

    def __init__(self, config: CommConfig) -> None:
        self._config = config
        self._initialized = False

    @property
    def config(self) -> CommConfig:
        """Return the communicator configuration."""
        return self._config

    @property
    def initialized(self) -> bool:
        """Return whether the process group has been initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the PyTorch distributed process group.

        Sets environment variables for rendezvous and creates the process
        group with the configured backend and timeout.

        Raises:
            RuntimeError: If initialization fails or process group already initialized.
        """
        if self._initialized:
            raise RuntimeError("Communicator already initialized")

        os.environ["MASTER_ADDR"] = self._config.master_addr
        os.environ["MASTER_PORT"] = str(self._config.master_port)
        os.environ["RANK"] = str(self._config.rank)
        os.environ["WORLD_SIZE"] = str(self._config.world_size)

        timeout = timedelta(seconds=self._config.timeout_seconds)

        logger.info(
            "Initializing distributed process group: "
            "backend=%s, rank=%d, world_size=%d, "
            "master=%s:%d, timeout=%ds, transport=%s",
            self._config.backend,
            self._config.rank,
            self._config.world_size,
            self._config.master_addr,
            self._config.master_port,
            self._config.timeout_seconds,
            self._config.transport,
        )

        dist.init_process_group(
            backend=self._config.backend,
            rank=self._config.rank,
            world_size=self._config.world_size,
            timeout=timeout,
        )

        self._initialized = True
        logger.info(
            "Process group initialized successfully: rank=%d/%d",
            self._config.rank,
            self._config.world_size,
        )

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Send a tensor to the destination rank.

        For Gloo backend, the tensor is moved to CPU before sending since
        Gloo does not support direct GPU tensor transfer. For NCCL, the
        tensor is sent directly from GPU memory.

        Args:
            tensor: The tensor to send.
            dst_rank: The rank of the destination process.

        Raises:
            RuntimeError: If the process group is not initialized.
            RuntimeError: If the send operation times out, with descriptive
                error identifying source rank, destination rank, and operation.
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized — call initialize() first")

        try:
            if self._config.backend == "gloo":
                cpu_tensor = tensor.cpu().contiguous()
                dist.send(cpu_tensor, dst=dst_rank)
            else:
                dist.send(tensor.contiguous(), dst=dst_rank)
        except RuntimeError as exc:
            if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
                raise RuntimeError(
                    f"Send operation timed out: "
                    f"src_rank={self._config.rank}, dst_rank={dst_rank}, "
                    f"operation=send, timeout={self._config.timeout_seconds}s"
                ) from exc
            raise

    def recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        target_device: str,
    ) -> torch.Tensor:
        """Receive a tensor from the source rank and move to target device.

        For Gloo backend, receives into a CPU tensor then moves to the
        target device. For NCCL, receives directly on the target device.

        Args:
            shape: Expected shape of the incoming tensor.
            dtype: Expected dtype of the incoming tensor.
            src_rank: The rank of the source process.
            target_device: Device string to place the received tensor on
                (e.g. "xpu:0", "cuda:0").

        Returns:
            The received tensor on the target device.

        Raises:
            RuntimeError: If the process group is not initialized.
            RuntimeError: If the recv operation times out, with descriptive
                error identifying source rank, destination rank, and operation.
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized — call initialize() first")

        try:
            if self._config.backend == "gloo":
                cpu_tensor = torch.empty(shape, dtype=dtype, device="cpu")
                dist.recv(cpu_tensor, src=src_rank)
                return cpu_tensor.to(target_device)
            else:
                gpu_tensor = torch.empty(shape, dtype=dtype, device=target_device)
                dist.recv(gpu_tensor, src=src_rank)
                return gpu_tensor
        except RuntimeError as exc:
            if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
                raise RuntimeError(
                    f"Recv operation timed out: "
                    f"src_rank={src_rank}, dst_rank={self._config.rank}, "
                    f"operation=recv, timeout={self._config.timeout_seconds}s"
                ) from exc
            raise

    def destroy(self) -> None:
        """Destroy the process group and release resources.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._initialized:
            logger.info("Destroying process group: rank=%d", self._config.rank)
            dist.destroy_process_group()
            self._initialized = False
