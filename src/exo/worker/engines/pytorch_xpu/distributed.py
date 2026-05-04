"""
PyTorch Distributed Communicator for Pipeline-Parallel Inference

Manages process group lifecycle and CPU-staged tensor communication using the
Gloo backend over TCP. Gloo is the only torch.distributed backend that works
across both NVIDIA CUDA and Intel XPU devices in a single process group.

CPU tensor staging pattern:
  - GPU → CPU before dist.send() (stage_to_cpu)
  - CPU → GPU after dist.recv() (unstage_from_cpu)

On Intel integrated GPU nodes (shared memory), the CPU↔GPU copy is nearly free.
On NVIDIA nodes, it crosses PCIe.

Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 2.1, 2.2, 2.3, 2.6, 2.7, 2a.1, 2a.4, 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessGroupConfig:
    """Configuration for torch.distributed process group initialization.

    The Gloo backend is used for all nodes regardless of GPU type, because
    XCCL is not supported on Intel Arc client GPUs and a single process group
    requires a single backend that works across both NVIDIA CUDA and Intel XPU.
    """

    rank: int
    world_size: int
    master_addr: str
    master_port: int
    backend: Literal["gloo"] = "gloo"
    init_timeout_seconds: int = 120


@dataclass(frozen=True)
class CpuStagedTensor:
    """A tensor that has been staged to CPU for Gloo transport.

    Gloo requires all tensors to be on CPU for send/recv operations.
    This dataclass preserves the original dtype and shape so the tensor
    can be correctly unstaged back to a GPU device after transport.
    """

    cpu_tensor: "torch.Tensor"
    original_dtype: "torch.dtype"
    original_shape: tuple[int, ...]


def init_process_group(config: ProcessGroupConfig) -> None:
    """Initialize torch.distributed with Gloo backend via env:// rendezvous.

    Sets MASTER_ADDR and MASTER_PORT environment variables, then calls
    torch.distributed.init_process_group with the Gloo backend. The env://
    init_method uses these environment variables for rendezvous through the
    aggregation switch.

    Requirements: 1.1, 1.3, 1.6
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    # Tell Gloo which network interface to use for mesh connections.
    # Without this, Gloo resolves the hostname which may point to a loopback
    # address (e.g., 127.0.0.2 in /etc/hosts on NixOS), causing connection failures.
    # We use psutil to find the network interface with a routable (non-loopback) IP.
    # Both GLOO_SOCKET_IFNAME and TP_SOCKET_IFNAME must be set — the former
    # controls rendezvous, the latter controls the actual data transport.
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        try:
            import psutil
            import socket
            detected_ifname: str | None = None

            # Strategy: find the interface whose IP is on the same subnet as MASTER_ADDR,
            # or failing that, the first non-loopback interface with a global IPv4 address.
            addrs = psutil.net_if_addrs()
            for ifname, if_addrs in addrs.items():
                if ifname == "lo":
                    continue
                for addr in if_addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                        # Check if this interface is on the same /24 as MASTER_ADDR
                        local_prefix = ".".join(addr.address.split(".")[:3])
                        master_prefix = ".".join(config.master_addr.split(".")[:3])
                        if local_prefix == master_prefix:
                            detected_ifname = ifname
                            break
                if detected_ifname:
                    break

            # Fallback: first non-loopback interface with any IPv4
            if detected_ifname is None:
                for ifname, if_addrs in addrs.items():
                    if ifname == "lo":
                        continue
                    for addr in if_addrs:
                        if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                            detected_ifname = ifname
                            break
                    if detected_ifname:
                        break

            if detected_ifname:
                os.environ["GLOO_SOCKET_IFNAME"] = detected_ifname
                os.environ["TP_SOCKET_IFNAME"] = detected_ifname
                logger.info(f"Set GLOO_SOCKET_IFNAME={detected_ifname}")
            else:
                logger.warning("Could not detect network interface for Gloo")
        except Exception as e:
            logger.warning(f"Failed to detect network interface for Gloo: {e}")

    logger.info(
        f"Gloo env: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}, "
        f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME', 'NOT SET')}, "
        f"TP_SOCKET_IFNAME={os.environ.get('TP_SOCKET_IFNAME', 'NOT SET')}"
    )

    try:
        dist.init_process_group(
            backend="gloo",
            rank=config.rank,
            world_size=config.world_size,
            init_method="env://",
            timeout=timedelta(seconds=config.init_timeout_seconds),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize process group: rank={config.rank}, "
            f"world_size={config.world_size}, backend=gloo, "
            f"master_addr={config.master_addr}, master_port={config.master_port}: {exc}"
        ) from exc


def destroy_process_group(timeout_seconds: float = 5.0) -> None:
    """Destroy process group with timeout to avoid blocking on unreachable peers.

    Uses a thread pool to enforce the timeout since
    torch.distributed.destroy_process_group() does not accept a timeout
    parameter. On timeout, logs a warning and proceeds without blocking.

    Requirements: 8.1, 8.2, 8.4
    """
    import torch.distributed as dist

    def _destroy() -> None:
        dist.destroy_process_group()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_destroy)
        try:
            future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            logger.warning(
                "destroy_process_group timed out after %.1f seconds, "
                "proceeding with local cleanup",
                timeout_seconds,
            )
        except Exception:
            logger.warning(
                "destroy_process_group raised an exception, "
                "proceeding with local cleanup",
                exc_info=True,
            )


def stage_to_cpu(tensor: "torch.Tensor") -> CpuStagedTensor:
    """Move a GPU tensor to CPU for Gloo transport.

    Preserves the original dtype and shape so the tensor can be correctly
    unstaged back to a GPU device after transport. On Intel integrated GPU
    nodes (shared memory architecture), this copy is nearly free.

    Requirements: 2.1, 2.2, 2.3
    """
    original_dtype = tensor.dtype
    original_shape = tuple(tensor.shape)
    cpu_tensor = tensor.to("cpu")
    return CpuStagedTensor(
        cpu_tensor=cpu_tensor,
        original_dtype=original_dtype,
        original_shape=original_shape,
    )


def unstage_from_cpu(staged: CpuStagedTensor, target_device: str) -> "torch.Tensor":
    """Move a CPU-staged tensor back to the target GPU device.

    Verifies that the dtype and shape match the original tensor metadata
    stored during staging.

    Requirements: 2.2, 2.3
    """
    result = staged.cpu_tensor.to(target_device)
    if result.dtype != staged.original_dtype:
        raise RuntimeError(
            f"Unstaged tensor dtype mismatch: expected {staged.original_dtype}, "
            f"got {result.dtype}"
        )
    if tuple(result.shape) != staged.original_shape:
        raise RuntimeError(
            f"Unstaged tensor shape mismatch: expected {staged.original_shape}, "
            f"got {tuple(result.shape)}"
        )
    return result


def send_activation(tensor: "torch.Tensor", dst_rank: int) -> None:
    """Stage tensor to CPU and send to destination rank via Gloo.

    The tensor is first moved to CPU (required by Gloo), then sent to the
    destination rank using torch.distributed.send.

    Requirements: 2.1, 2.6, 2a.2
    """
    import torch.distributed as dist

    staged = stage_to_cpu(tensor)
    try:
        dist.send(staged.cpu_tensor, dst=dst_rank)
    except Exception as exc:
        src_rank = dist.get_rank()
        raise RuntimeError(
            f"Failed to send activation: src_rank={src_rank}, dst_rank={dst_rank}, "
            f"tensor_shape={staged.original_shape}: {exc}"
        ) from exc


def recv_activation(
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    src_rank: int,
    target_device: str,
) -> "torch.Tensor":
    """Receive tensor on CPU buffer and move to target GPU device.

    Allocates a CPU buffer with the specified shape and dtype, receives data
    from the source rank via Gloo, then unstages the tensor to the target
    GPU device.

    Requirements: 2.2, 2.6, 2a.2
    """
    import torch
    import torch.distributed as dist

    buffer = torch.empty(shape, dtype=dtype, device="cpu")
    try:
        dist.recv(buffer, src=src_rank)
    except Exception as exc:
        local_rank = dist.get_rank()
        raise RuntimeError(
            f"Failed to receive activation: src_rank={src_rank}, "
            f"local_rank={local_rank}, tensor_shape={shape}: {exc}"
        ) from exc

    staged = CpuStagedTensor(
        cpu_tensor=buffer,
        original_dtype=dtype,
        original_shape=shape,
    )
    return unstage_from_cpu(staged, target_device)
