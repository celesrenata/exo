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
class TensorParallelGroupConfig:
    """Configuration for tensor-parallel process group over TB4.

    Separate from ProcessGroupConfig (used for pipeline parallelism over ethernet).
    The tensor-parallel group uses Gloo over Thunderbolt 4 network interfaces,
    which provide 40 Gbps bandwidth for the frequent all-reduce operations
    that tensor parallelism requires.

    Requirements: 4.1, 9.1, 9.5
    """

    rank: int
    world_size: int
    master_addr: str  # TB4 IP of rank 0
    master_port: int  # Ephemeral port for TP group
    tb4_interface_name: str  # Interface name for GLOO_SOCKET_IFNAME
    backend: Literal["gloo"] = "gloo"
    init_timeout_seconds: int = 60  # Shorter than ethernet — TB4 is local
    allreduce_timeout_seconds: int = 30


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
    """Initialize torch.distributed with Gloo backend using TCPStore.

    Creates a TCPStore directly (not env:// rendezvous) so that rank 0
    can bind on 0.0.0.0 while other ranks connect to master_addr.

    Requirements: 1.1, 1.3, 1.6
    """
    import torch.distributed as dist

    # Tell Gloo which network interface to use for mesh connections.
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        try:
            import fcntl
            import socket
            import struct
            detected_ifname: str | None = None

            with open("/proc/net/dev") as f:
                lines = f.readlines()[2:]

            master_prefix = ".".join(config.master_addr.split(".")[:3])

            for line in lines:
                ifname = line.split(":")[0].strip()
                if ifname == "lo" or ifname.startswith("veth") or ifname.startswith("docker") or ifname.startswith("cni") or ifname.startswith("flannel"):
                    continue
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    ip_bytes = fcntl.ioctl(
                        sock.fileno(),
                        0x8915,
                        struct.pack("256s", ifname.encode("utf-8")[:15])
                    )[20:24]
                    ip_addr = socket.inet_ntoa(ip_bytes)
                    sock.close()

                    if ip_addr.startswith(master_prefix):
                        detected_ifname = ifname
                        break
                except (OSError, struct.error):
                    continue

            if detected_ifname:
                os.environ["GLOO_SOCKET_IFNAME"] = detected_ifname
                os.environ["TP_SOCKET_IFNAME"] = detected_ifname
                logger.info(f"Set GLOO_SOCKET_IFNAME={detected_ifname}")
        except Exception as e:
            logger.warning(f"Failed to detect network interface: {e}")

    # Use TCPStore directly — the node whose bond0 IP matches master_addr
    # hosts the store. This ensures gremlin-1 (MASTER_ADDR=10.1.1.12) always
    # hosts the TCPStore when it's part of the instance.
    import socket as _socket
    try:
        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        _s.connect(("10.1.1.1", 1))
        _our_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _our_ip = ""

    is_master = (_our_ip == config.master_addr)
    store_host = "0.0.0.0" if is_master else config.master_addr

    logger.info(
        f"TCPStore: host={store_host}, port={config.master_port}, "
        f"is_master={is_master}, our_ip={_our_ip}, rank={config.rank}, "
        f"master_addr={config.master_addr}"
    )

    store = dist.TCPStore(
        host_name=store_host,
        port=config.master_port,
        world_size=config.world_size,
        is_master=is_master,
        timeout=timedelta(seconds=config.init_timeout_seconds),
    )

    dist.init_process_group(
        backend=config.backend,
        store=store,
        rank=config.rank,
        world_size=config.world_size,
        timeout=timedelta(seconds=config.init_timeout_seconds),
    )

    logger.info(
        f"Process group initialized: rank={config.rank}/{config.world_size}, "
        f"backend={config.backend}, master={config.master_addr}:{config.master_port}, "
        f"is_master={is_master}"
    )


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


# Module-level handle for the tensor-parallel process group.
# None means no TP group has been initialized yet.
_tp_process_group: object | None = None


def get_tensor_parallel_group() -> object | None:
    """Return the tensor-parallel process group handle, or None if not initialized."""
    return _tp_process_group


def init_tensor_parallel_group(config: TensorParallelGroupConfig) -> None:
    """Initialize a Gloo process group for tensor parallelism over TB4.

    Sets GLOO_SOCKET_IFNAME to the TB4 interface so all collective
    operations route over the high-bandwidth TB4 links.

    If a default process group already exists (e.g., pipeline parallelism
    over ethernet), this creates a new named group that coexists with it.
    If no default group exists, this initializes the default group directly.

    After initialization, the process group handle is stored in the module-level
    _tp_process_group variable, accessible via get_tensor_parallel_group().

    Requirements: 4.1, 4.4, 4.6, 9.1, 9.6
    """
    import torch.distributed as dist

    global _tp_process_group  # noqa: PLW0603

    # Set GLOO_SOCKET_IFNAME to the TB4 interface so Gloo binds to the
    # high-bandwidth Thunderbolt 4 link instead of ethernet.
    os.environ["GLOO_SOCKET_IFNAME"] = config.tb4_interface_name
    os.environ["TP_SOCKET_IFNAME"] = config.tb4_interface_name

    # Set env:// rendezvous variables for the TB4 master.
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["RANK"] = str(config.rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)

    logger.info(
        f"Initializing tensor-parallel process group: "
        f"rank={config.rank}, world_size={config.world_size}, "
        f"master_addr={config.master_addr}, master_port={config.master_port}, "
        f"tb4_interface={config.tb4_interface_name}, "
        f"timeout={config.init_timeout_seconds}s"
    )

    timeout = timedelta(seconds=config.init_timeout_seconds)

    if dist.is_initialized():
        # A default process group already exists (pipeline parallelism).
        # Create a separate group for tensor parallelism that coexists.
        logger.info(
            "Default process group already initialized (pipeline parallelism). "
            "Creating new group for tensor parallelism."
        )
        try:
            ranks = list(range(config.world_size))
            _tp_process_group = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create tensor-parallel process group: "
                f"rank={config.rank}, world_size={config.world_size}, "
                f"master_addr={config.master_addr}, "
                f"tb4_interface={config.tb4_interface_name}: {exc}"
            ) from exc
    else:
        # No default group exists. Initialize as the default process group.
        try:
            dist.init_process_group(
                backend="gloo",
                rank=config.rank,
                world_size=config.world_size,
                init_method="env://",
                timeout=timeout,
            )
            # The default group is the TP group in this case.
            _tp_process_group = dist.group.WORLD
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize tensor-parallel process group: "
                f"rank={config.rank}, world_size={config.world_size}, "
                f"master_addr={config.master_addr}, master_port={config.master_port}, "
                f"tb4_interface={config.tb4_interface_name}: {exc}"
            ) from exc

    logger.info(
        f"Tensor-parallel process group initialized: "
        f"rank={config.rank}, world_size={config.world_size}, "
        f"GLOO_SOCKET_IFNAME={config.tb4_interface_name}"
    )


def verify_tensor_parallel_group(world_size: int) -> bool:
    """Verify TP group connectivity with a test all-reduce.

    Each rank contributes its rank ID. The sum should equal
    world_size * (world_size - 1) / 2.

    Returns True if verification passes, False otherwise.

    Requirements: 9.3
    """
    import torch
    import torch.distributed as dist

    try:
        rank = dist.get_rank()
        tensor = torch.tensor([rank], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=get_tensor_parallel_group())

        expected = world_size * (world_size - 1) // 2
        result = tensor.item()
        if result == expected:
            logger.info(
                f"Tensor-parallel group verification passed: "
                f"rank={rank}, all_reduce(rank_ids)={result}, expected={expected}"
            )
            return True
        else:
            logger.error(
                f"Tensor-parallel group verification FAILED: "
                f"rank={rank}, all_reduce(rank_ids)={result}, expected={expected}"
            )
            return False
    except Exception as exc:
        logger.error(
            f"Tensor-parallel group verification failed with exception: {exc}",
            exc_info=True,
        )
        return False


def derive_rank_assignment(node_ips: list[str]) -> dict[str, int]:
    """Derive consistent rank assignments from a list of node IPs.

    Sorts node IPs lexicographically and assigns rank 0 to the first IP
    (which becomes MASTER_ADDR). This ensures all nodes derive the same
    rank assignment regardless of which node performs the derivation.

    Args:
        node_ips: List of node IP addresses participating in the group.

    Returns:
        Mapping of node_ip → rank (0-indexed).

    Raises:
        ValueError: If node_ips is empty or contains duplicates.

    Requirements: 9.2
    """
    if not node_ips:
        raise ValueError("node_ips must not be empty")
    if len(node_ips) != len(set(node_ips)):
        raise ValueError(
            f"node_ips contains duplicates: {node_ips}"
        )
    sorted_ips = sorted(node_ips)
    return {ip: rank for rank, ip in enumerate(sorted_ips)}


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
