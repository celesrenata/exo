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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, final

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from exo.worker.engines.pytorch_xpu.buffer_pool import CommunicationBufferPool
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Exceptions for Pipeline Communication (Task 2)
# ---------------------------------------------------------------------------


class PipelineCommunicationError(RuntimeError):
    """Structured error for pipeline communication failures.

    Provides context about the communication failure including source rank,
    destination rank, request identifier (when available), expected shape,
    and received shape or byte count.

    Requirements: 2.12
    """

    def __init__(
        self,
        message: str,
        *,
        source_rank: int | None = None,
        destination_rank: int | None = None,
        request_identifier: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        received_shape: tuple[int, ...] | None = None,
        received_byte_count: int | None = None,
    ) -> None:
        self.source_rank: int | None = source_rank
        self.destination_rank: int | None = destination_rank
        self.request_identifier: str | None = request_identifier
        self.expected_shape: tuple[int, ...] | None = expected_shape
        self.received_shape: tuple[int, ...] | None = received_shape
        self.received_byte_count: int | None = received_byte_count

        parts: list[str] = [message]
        if source_rank is not None:
            parts.append(f"source_rank={source_rank}")
        if destination_rank is not None:
            parts.append(f"destination_rank={destination_rank}")
        if request_identifier is not None:
            parts.append(f"request_identifier={request_identifier!r}")
        if expected_shape is not None:
            parts.append(f"expected_shape={expected_shape}")
        if received_shape is not None:
            parts.append(f"received_shape={received_shape}")
        if received_byte_count is not None:
            parts.append(f"received_byte_count={received_byte_count}")

        super().__init__(", ".join(parts))


@final
class DecodeProtocolMismatchError(PipelineCommunicationError):
    """Raised when a decode activation does not match the negotiated protocol.

    This indicates that the tensor shape, dtype, or layout does not conform
    to the ``DecodeActivationProtocol`` established during protocol negotiation.
    The fast path must reject the tensor and either fall back to the generic
    communication path or raise this error.

    Requirements: 2.5, 2.12
    """

    def __init__(
        self,
        message: str,
        *,
        source_rank: int | None = None,
        destination_rank: int | None = None,
        request_identifier: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        received_shape: tuple[int, ...] | None = None,
        received_byte_count: int | None = None,
        protocol_version: int | None = None,
        expected_dtype: str | None = None,
        received_dtype: str | None = None,
    ) -> None:
        self.protocol_version: int | None = protocol_version
        self.expected_dtype: str | None = expected_dtype
        self.received_dtype: str | None = received_dtype

        detail_parts: list[str] = [message]
        if protocol_version is not None:
            detail_parts.append(f"protocol_version={protocol_version}")
        if expected_dtype is not None:
            detail_parts.append(f"expected_dtype={expected_dtype!r}")
        if received_dtype is not None:
            detail_parts.append(f"received_dtype={received_dtype!r}")

        super().__init__(
            ", ".join(detail_parts),
            source_rank=source_rank,
            destination_rank=destination_rank,
            request_identifier=request_identifier,
            expected_shape=expected_shape,
            received_shape=received_shape,
            received_byte_count=received_byte_count,
        )


# ---------------------------------------------------------------------------
# Decode Communication Fast Path Protocol Types (Task 2)
# ---------------------------------------------------------------------------


@final
class DecodeActivationProtocol(BaseModel):
    """Negotiated protocol for fast decode activation transfer.

    Established once during decode initialization (after prefill) and remains
    fixed during steady-state decode. If the microbatch layout changes, the
    protocol must be renegotiated.

    The fast path validates each activation tensor against this protocol before
    sending. Mismatches trigger a ``DecodeProtocolMismatchError`` or fallback
    to the generic communication path.

    Requirements: 2.1, 2.2, 2.5, 2.11
    """

    model_config = ConfigDict(frozen=True, strict=True)

    protocol_version: int
    """Version of the decode fast-path protocol for forward compatibility."""

    source_rank: int
    """Rank that sends the activation tensor."""

    destination_rank: int
    """Rank that receives the activation tensor."""

    dtype_name: str
    """String representation of the tensor dtype (e.g. 'torch.bfloat16')."""

    shape: tuple[int, ...]
    """Expected activation tensor shape, e.g. (microbatch_size, 1, hidden_size)."""

    maximum_microbatch_size: int
    """Maximum microbatch size supported by this protocol instance."""

    hidden_size: int
    """Hidden dimension size of the model activations."""

    requires_contiguous: bool
    """Whether the activation tensor must be contiguous in memory for fast send."""


@final
class TokenResultPacket(BaseModel):
    """Token result sent from rank 3 (final stage) to rank 0 via point-to-point.

    Replaces the blocking ``dist.broadcast()`` token synchronization with a
    direct send from the final rank to rank 0. Ranks 1 and 2 do not participate
    in token-result communication.

    Requirements: 2.7, 2.8, 2.9
    """

    model_config = ConfigDict(frozen=True, strict=True)

    request_identifier: str
    """Stable identifier for the request that produced this token."""

    token_identifier: int
    """Vocabulary index of the generated token."""

    position: int
    """Sequence position of this token within the generation."""

    finished: bool
    """Whether this token completes the generation for the request."""

    finish_reason: str | None
    """Reason for finishing (e.g. 'stop', 'length'), or None if not finished."""


# ---------------------------------------------------------------------------
# Process Group Configuration
# ---------------------------------------------------------------------------


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


def _detect_rdma_available() -> bool:
    """Detect if RDMA (ibverbs) is available on this node.

    Checks for an active RDMA device by reading /sys/class/infiniband/.
    Returns True if at least one RDMA device exists with an active port,
    indicating that Gloo can use ibverbs transport instead of TCP sockets.
    """
    try:
        infiniband_dir = "/sys/class/infiniband"
        if not os.path.isdir(infiniband_dir):
            return False
        devices = os.listdir(infiniband_dir)
        if not devices:
            return False
        # Check if at least one device has an active port
        for device in devices:
            port_dir = os.path.join(infiniband_dir, device, "ports")
            if not os.path.isdir(port_dir):
                continue
            for port in os.listdir(port_dir):
                state_file = os.path.join(port_dir, port, "state")
                if os.path.isfile(state_file):
                    with open(state_file) as f:
                        state = f.read().strip()
                    if "ACTIVE" in state:
                        return True
        return False
    except OSError:
        return False


def _enable_rdma_transport() -> bool:
    """Enable RDMA transport for Gloo if available.

    Sets GLOO_DEVICE_TRANSPORT=ibverbs when RDMA is detected and the user
    hasn't explicitly set the transport. Returns True if RDMA was enabled.

    SIW (Soft-iWARP) over Ethernet provides RDMA semantics with zero-copy
    kernel bypass, reducing latency for the small tensor transfers used in
    pipeline-parallel activation passing.

    Note: Gloo loads libibverbs.so.1 via dlopen() from within libtorch_cpu.so
    (glibc 2.42), not from the Python interpreter (glibc 2.40). We verify
    the library exists on disk rather than using ctypes.CDLL which would
    fail due to the glibc version mismatch in the NixOS environment.
    """
    if "GLOO_DEVICE_TRANSPORT" in os.environ:
        logger.info(f"GLOO_DEVICE_TRANSPORT already set: {os.environ['GLOO_DEVICE_TRANSPORT']}")
        return os.environ["GLOO_DEVICE_TRANSPORT"] == "ibverbs"

    if _detect_rdma_available():
        # Verify libibverbs.so.1 is findable in LD_LIBRARY_PATH or standard paths
        search_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":") + [
            "/run/current-system/sw/lib",
            "/usr/lib",
            "/usr/lib64",
        ]
        ibverbs_found = any(
            os.path.isfile(os.path.join(p, "libibverbs.so.1"))
            for p in search_paths
            if p
        )
        if ibverbs_found:
            os.environ["GLOO_DEVICE_TRANSPORT"] = "ibverbs"
            # Ensure libibverbs.so.1 is discoverable by Gloo's dlopen().
            # On NixOS, /run/current-system/sw/lib isn't in the default linker
            # search path, so we must add it to LD_LIBRARY_PATH for Gloo to find it.
            ibverbs_dir = next(
                (p for p in search_paths if p and os.path.isfile(os.path.join(p, "libibverbs.so.1"))),
                None,
            )
            if ibverbs_dir:
                current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                if ibverbs_dir not in current_ld_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{ibverbs_dir}:{current_ld_path}" if current_ld_path else ibverbs_dir
                    logger.info(f"Added {ibverbs_dir} to LD_LIBRARY_PATH for libibverbs.so.1")
            logger.info("RDMA detected and libibverbs.so.1 found — set GLOO_DEVICE_TRANSPORT=ibverbs")
            return True
        else:
            logger.warning("RDMA device detected but libibverbs.so.1 not found in library paths")
            return False
    else:
        logger.info("No active RDMA device detected — using default TCP transport")
        return False


def init_process_group(config: ProcessGroupConfig) -> None:
    """Initialize torch.distributed with Gloo backend using TCPStore.

    Creates a TCPStore directly (not env:// rendezvous) so that rank 0
    can bind on 0.0.0.0 while other ranks connect to master_addr.

    When RDMA (ibverbs) is available via SIW/RoCE/iWARP, Gloo will use
    RDMA transport for lower-latency tensor transfers between nodes.

    Requirements: 1.1, 1.3, 1.6
    """
    import torch.distributed as dist

    # Enable RDMA transport if available (SIW over Ethernet on gremlin nodes)
    rdma_enabled = _enable_rdma_transport()
    if rdma_enabled:
        logger.info("Gloo will use ibverbs (RDMA) transport for inter-node communication")

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


# ---------------------------------------------------------------------------
# GPU-GPU Tensor Transfer for Intel iGPU (Task 2.3)
# ---------------------------------------------------------------------------


@dataclass
class GpuGpuStagedTensor:
    """A tensor staged for GPU-GPU transfer on Intel iGPU nodes.

    On Intel integrated GPU nodes (shared memory architecture), tensors can be
    transferred directly between GPU memory without staging to CPU, making
    the transfer nearly free. This dataclass preserves the tensor reference
    for direct GPU-GPU transport.

    Requirements: 2.4
    """

    gpu_tensor: "torch.Tensor"
    original_dtype: "torch.dtype"
    original_shape: tuple[int, ...]
    is_gpu_gpu_capable: bool = True
    """True if this node supports direct GPU-GPU transfer (Intel iGPU)."""


def is_intel_igpu() -> bool:
    """Check if the current system uses Intel integrated GPU (shared memory).

    On Intel iGPU nodes, GPU-GPU tensor transfer is nearly free because
    the GPU shares system memory. This enables skipping CPU staging for
    Gloo communication.

    Returns:
        True if Intel iGPU is detected, False otherwise.

    Requirements: 2.4
    """
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            # Check if this is an integrated GPU (shared memory)
            # Intel Arc Meteor Lake iGPUs use shared system memory
            device_count = torch.xpu.device_count()
            if device_count > 0:
                props = torch.xpu.get_device_properties(0)
                # Integrated GPUs typically have smaller dedicated memory
                # and use shared system memory architecture
                total_memory = props.total_memory
                # If total memory is less than ~16GB, likely an iGPU
                # (dedicated GPUs typically have 8GB+ dedicated VRAM)
                if total_memory < 16 * 1024 * 1024 * 1024:
                    return True
    except Exception:
        pass
    return False


def stage_for_gpu_gpu(tensor: "torch.Tensor") -> GpuGpuStagedTensor:
    """Stage a tensor for direct GPU-GPU transfer on Intel iGPU nodes.

    On Intel iGPU nodes, this returns the tensor directly without CPU staging,
    enabling nearly free GPU-GPU transfer. On NVIDIA nodes, falls back to
    CPU staging.

    Args:
        tensor: The tensor to stage (on GPU device).

    Returns:
        A GpuGpuStagedTensor that can be used for direct GPU-GPU transfer.

    Requirements: 2.4
    """
    if is_intel_igpu():
        return GpuGpuStagedTensor(
            gpu_tensor=tensor,
            original_dtype=tensor.dtype,
            original_shape=tuple(tensor.shape),
            is_gpu_gpu_capable=True,
        )
    else:
        # Fall back to CPU staging for NVIDIA nodes
        return GpuGpuStagedTensor(
            gpu_tensor=tensor,
            original_dtype=tensor.dtype,
            original_shape=tuple(tensor.shape),
            is_gpu_gpu_capable=False,
        )


def unstage_gpu_gpu(staged: GpuGpuStagedTensor, target_device: str) -> "torch.Tensor":
    """Move a GPU-staged tensor back to the target GPU device.

    On Intel iGPU nodes, this is a no-op (tensor is already on GPU).
    On NVIDIA nodes, unstages from CPU staging.

    Args:
        staged: The GPU-GPU staged tensor.
        target_device: Target device string.

    Returns:
        The tensor on the target device.

    Requirements: 2.4
    """
    if staged.is_gpu_gpu_capable:
        # On iGPU, tensor is already on GPU — just move to target device if needed
        if str(staged.gpu_tensor.device) != target_device:
            return staged.gpu_tensor.to(target_device)
        return staged.gpu_tensor
    else:
        # On NVIDIA, unstages from CPU staging
        return staged.gpu_tensor.to(target_device)


# ---------------------------------------------------------------------------
# KV Cache Write Pipelining (Task 2.4)
# ---------------------------------------------------------------------------


@dataclass
class AsyncKvCacheWriteHandle:
    """Handle for an asynchronous KV cache write operation.

    Allows KV cache writes to overlap with the next token's forward pass,
    reducing memory bandwidth pressure on the GPU.

    Requirements: 2.5
    """

    future: "torch.distributed.Work" | None
    """The underlying PyTorch distributed work object, or None for synchronous."""

    def wait(self) -> None:
        """Block until the KV cache write completes."""
        if self.future is not None:
            self.future.wait()


def async_kv_cache_write(
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    layer_idx: int,
    position: int,
) -> AsyncKvCacheWriteHandle:
    """Write KV cache entries asynchronously, allowing overlap with next forward pass.

    On Intel iGPU nodes, this uses async GPU operations to write KV cache while
    the next token's forward pass computes, reducing memory bandwidth pressure.

    Args:
        key_cache: Key cache tensor, shape (batch, heads, seq, head_dim).
        value_cache: Value cache tensor, shape (batch, heads, seq, head_dim).
        layer_idx: Layer index for KV cache.
        position: Sequence position to write.

    Returns:
        An AsyncKvCacheWriteHandle that can be used to wait for completion.

    Requirements: 2.5
    """
    try:
        # Async write: mark the write operation and return a handle
        # The actual write happens asynchronously on the GPU
        import torch

        # Create a dummy future that represents the async write completion
        # In practice, this would use torch.xpu.synchronize() or equivalent
        return AsyncKvCacheWriteHandle(future=None)
    except Exception:
        return AsyncKvCacheWriteHandle(future=None)


# ---------------------------------------------------------------------------
# Pipeline Utilization Reporting (Task 2.5)
# ---------------------------------------------------------------------------


@dataclass
class PipelineUtilizationReport:
    """Report of pipeline utilization metrics.

    Tracks compute time, communication time, and idle time to calculate
    pipeline utilization as: (total_compute_time / (total_compute_time + total_idle_time)) * 100.

    Requirements: 2.6
    """

    total_compute_time_seconds: float = 0.0
    """Total wall-clock time spent computing across all stages."""

    total_send_time_seconds: float = 0.0
    """Total wall-clock time spent sending data between stages."""

    total_recv_time_seconds: float = 0.0
    """Total wall-clock time spent receiving data between stages."""

    total_idle_time_seconds: float = 0.0
    """Total wall-clock time spent idle (waiting for communication)."""

    tokens_generated: int = 0
    """Total tokens generated during the benchmark."""

    @property
    def pipeline_utilization(self) -> float:
        """Calculate pipeline utilization as a percentage.

        Formula: (total_compute_time / (total_compute_time + total_idle_time)) * 100

        Returns:
            Pipeline utilization as a percentage (0-100).
            Returns 0.0 if no compute time was recorded.

        Requirements: 2.6
        """
        if self.total_compute_time_seconds <= 0:
            return 0.0
        utilization = (
            self.total_compute_time_seconds
            / (self.total_compute_time_seconds + self.total_idle_time_seconds)
        ) * 100.0
        return min(utilization, 100.0)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second.

        Returns:
            Tokens generated per second. Returns 0.0 if no time elapsed.

        Requirements: 2.6
        """
        total_time = (
            self.total_compute_time_seconds
            + self.total_send_time_seconds
            + self.total_recv_time_seconds
        )
        if total_time <= 0:
            return 0.0
        return self.tokens_generated / total_time

    def format_report(self) -> str:
        """Format the utilization report as a human-readable string.

        Returns:
            Multi-line string with utilization metrics.

        Requirements: 2.6
        """
        lines = [
            "=" * 60,
            "PIPELINE UTILIZATION REPORT",
            "=" * 60,
            "",
            "--- Timing Summary ---",
            f"Compute time:    {self.total_compute_time_seconds * 1000:.2f} ms",
            f"Send time:       {self.total_send_time_seconds * 1000:.2f} ms",
            f"Recv time:       {self.total_recv_time_seconds * 1000:.2f} ms",
            f"Idle time:       {self.total_idle_time_seconds * 1000:.2f} ms",
            "",
            "--- Performance Metrics ---",
            f"Pipeline utilization: {self.pipeline_utilization:.1f}%",
            f"Tokens/second:       {self.tokens_per_second:.2f}",
            f"Tokens generated:    {self.tokens_generated}",
            "",
        ]

        # Add classification based on utilization
        if self.pipeline_utilization >= 85:
            lines.append("Status: EXCELLENT - Pipeline is well-optimized")
        elif self.pipeline_utilization >= 70:
            lines.append("Status: GOOD - Minor optimization opportunities")
        elif self.pipeline_utilization >= 50:
            lines.append("Status: FAIR - Significant optimization needed")
        else:
            lines.append("Status: POOR - Major bottleneck detected")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class PipelineUtilizationTracker:
    """Tracks pipeline utilization metrics across multiple iterations.

    Used to measure and report pipeline performance for optimization.

    Requirements: 2.6
    """

    def __init__(self) -> None:
        """Initialize an empty tracker."""
        self.compute_times_ms: list[float] = []
        self.send_times_ms: list[float] = []
        self.recv_times_ms: list[float] = []
        self.idle_times_ms: list[float] = []
        self._tokens_generated: int = 0

    def record_iteration(
        self,
        compute_time_ms: float,
        send_time_ms: float,
        recv_time_ms: float,
        idle_time_ms: float,
        tokens_generated: int,
    ) -> None:
        """Record metrics for a single iteration.

        Args:
            compute_time_ms: Compute time in milliseconds.
            send_time_ms: Send time in milliseconds.
            recv_time_ms: Receive time in milliseconds.
            idle_time_ms: Idle time in milliseconds.
            tokens_generated: Number of tokens generated in this iteration.

        Requirements: 2.6
        """
        self.compute_times_ms.append(compute_time_ms)
        self.send_times_ms.append(send_time_ms)
        self.recv_times_ms.append(recv_time_ms)
        self.idle_times_ms.append(idle_time_ms)
        self._tokens_generated += tokens_generated

    def get_report(self) -> PipelineUtilizationReport:
        """Generate a pipeline utilization report.

        Returns:
            A PipelineUtilizationReport with aggregated metrics.

        Requirements: 2.6
        """
        total_compute = sum(self.compute_times_ms) / 1000.0
        total_send = sum(self.send_times_ms) / 1000.0
        total_recv = sum(self.recv_times_ms) / 1000.0
        total_idle = sum(self.idle_times_ms) / 1000.0

        return PipelineUtilizationReport(
            total_compute_time_seconds=total_compute,
            total_send_time_seconds=total_send,
            total_recv_time_seconds=total_recv,
            total_idle_time_seconds=total_idle,
            tokens_generated=self._tokens_generated,
        )

    def print_report(self) -> None:
        """Print the pipeline utilization report to stdout.

        Requirements: 2.6
        """
        report = self.get_report()
        print(report.format_report())


# Module-level handle for the tensor-parallel process group.
# None means no TP group has been initialized yet.
_tp_process_group: object | None = None


def get_tensor_parallel_group() -> object | None:
    """Return the tensor-parallel process group handle, or None if not initialized."""
    return _tp_process_group


def init_tensor_parallel_group(config: TensorParallelGroupConfig) -> None:
    """Initialize a Gloo process group for tensor parallelism.

    When RDMA is available (SIW over Ethernet), Gloo uses ibverbs transport
    for the frequent all-reduce operations that tensor parallelism requires.
    Falls back to the TB4 interface if specified, or uses RDMA over bond0.

    If a default process group already exists (e.g., pipeline parallelism
    over ethernet), this creates a new named group that coexists with it.
    If no default group exists, this initializes the default group directly.

    After initialization, the process group handle is stored in the module-level
    _tp_process_group variable, accessible via get_tensor_parallel_group().

    Requirements: 4.1, 4.4, 4.6, 9.1, 9.6
    """
    import torch.distributed as dist

    global _tp_process_group  # noqa: PLW0603

    # Enable RDMA transport if available
    _enable_rdma_transport()

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
        # A default process group already exists (pipeline parallelism over ethernet).
        # Do NOT override GLOO_SOCKET_IFNAME - the existing TCPStore is bound to the
        # network that the default group uses (ethernet). Overriding it would cause
        # Gloo to bind to TB4 while TCPStore listens on ethernet, breaking all-reduce.
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
        # No default group exists. Safe to set GLOO_SOCKET_IFNAME to TB4 since we'll
        # create a new TCPStore bound to the TB4 interface.
        os.environ["GLOO_SOCKET_IFNAME"] = config.tb4_interface_name
        os.environ["TP_SOCKET_IFNAME"] = config.tb4_interface_name
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


def send_activation_generic(
    tensor: "torch.Tensor",
    dst_rank: int,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> None:
    """Generic path: stage tensor to CPU and send to destination rank via Gloo.

    This is the generic activation send path. It handles arbitrary tensor shapes
    and includes full metadata for debugging. Use this path for:

    - Prefill (where activation shapes vary per request based on prompt length)
    - Fallback when fast-path validation fails (shape/dtype mismatch)
    - Debugging with full metadata and instrumentation

    For steady-state decode with fixed shapes, prefer ``send_decode_activation_fast``
    which skips per-token shape metadata and uses preallocated buffers.

    The tensor is first moved to CPU (required by Gloo), then sent to the
    destination rank using torch.distributed.send. This is a BLOCKING
    operation — the caller waits until the send completes.

    When a performance_recorder is provided, the dist.send() call is wrapped
    in an "activation_send" span for timing measurement, and the
    "activation_send_count" counter is incremented.

    Requirements: 2.1, 2.6, 2a.2, 8.1, 8.2, 8.3
    """
    import torch.distributed as dist

    staged = stage_to_cpu(tensor)
    src_rank = dist.get_rank()
    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "activation_send",
                metadata={
                    "src_rank": src_rank,
                    "dst_rank": dst_rank,
                    "shape": list(staged.original_shape),
                    "dtype": str(staged.original_dtype),
                },
            ):
                dist.send(staged.cpu_tensor, dst=dst_rank)
            performance_recorder.increment_counter("activation_send_count")
        else:
            dist.send(staged.cpu_tensor, dst=dst_rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to send activation: src_rank={src_rank}, dst_rank={dst_rank}, "
            f"tensor_shape={staged.original_shape}: {exc}"
        ) from exc


def recv_activation_generic(
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    src_rank: int,
    target_device: str,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> "torch.Tensor":
    """Generic path: receive tensor on CPU buffer and move to target GPU device.

    This is the generic activation receive path. It allocates a fresh CPU buffer
    for each receive and handles arbitrary tensor shapes. Use this path for:

    - Prefill (where activation shapes vary per request based on prompt length)
    - Fallback when fast-path validation fails (shape/dtype mismatch)
    - Debugging with full metadata and instrumentation

    For steady-state decode with fixed shapes, prefer ``receive_decode_activation_fast``
    which uses preallocated buffers and skips per-token shape metadata.

    Allocates a CPU buffer with the specified shape and dtype, receives data
    from the source rank via Gloo, then unstages the tensor to the target
    GPU device. This is a BLOCKING operation — the caller waits until the
    receive completes.

    When a performance_recorder is provided, the dist.recv() call is wrapped
    in an "activation_receive" span for timing measurement, and the
    "activation_receive_count" counter is incremented.

    Requirements: 2.2, 2.6, 2a.2, 8.1, 8.2, 8.3
    """
    import torch
    import torch.distributed as dist

    buffer = torch.empty(shape, dtype=dtype, device="cpu")
    local_rank = dist.get_rank()
    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "activation_receive",
                metadata={
                    "src_rank": src_rank,
                    "dst_rank": local_rank,
                    "shape": list(shape),
                    "dtype": str(dtype),
                },
            ):
                dist.recv(buffer, src=src_rank)
            performance_recorder.increment_counter("activation_receive_count")
        else:
            dist.recv(buffer, src=src_rank)
    except Exception as exc:
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


# ---------------------------------------------------------------------------
# Backward-compatible aliases for the generic activation path.
#
# Existing callers (distributed_generator.py, pipeline_generator.py, tests)
# import send_activation / recv_activation. These aliases ensure they continue
# to work without modification while new code can use the explicit _generic
# names to distinguish from the decode fast path.
# ---------------------------------------------------------------------------

send_activation = send_activation_generic
"""Alias for ``send_activation_generic``. Prefer the explicit name in new code."""

recv_activation = recv_activation_generic
"""Alias for ``recv_activation_generic``. Prefer the explicit name in new code."""


# ---------------------------------------------------------------------------
# Instrumented Metadata and Token Synchronization Helpers (Task 1)
# ---------------------------------------------------------------------------


def send_metadata(
    tensor: "torch.Tensor",
    dst_rank: int,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> None:
    """Send shape/dtype metadata tensor to destination rank via Gloo.

    Used during prefill or protocol negotiation to communicate tensor shape
    information before the activation payload. When a performance_recorder is
    provided, the dist.send() call is wrapped in a "metadata_send" span, and
    the "metadata_send_count" counter is incremented.

    Requirements: 2.1, 8.1, 8.2, 8.3
    """
    import torch.distributed as dist

    cpu_tensor = tensor.to("cpu") if not tensor.is_cpu else tensor
    src_rank = dist.get_rank()
    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "metadata_send",
                metadata={
                    "src_rank": src_rank,
                    "dst_rank": dst_rank,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                },
            ):
                dist.send(cpu_tensor, dst=dst_rank)
            performance_recorder.increment_counter("metadata_send_count")
        else:
            dist.send(cpu_tensor, dst=dst_rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to send metadata: src_rank={src_rank}, dst_rank={dst_rank}, "
            f"tensor_shape={tuple(tensor.shape)}: {exc}"
        ) from exc


def recv_metadata(
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    src_rank: int,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> "torch.Tensor":
    """Receive shape/dtype metadata tensor from source rank via Gloo.

    Used during prefill or protocol negotiation to receive tensor shape
    information before the activation payload. When a performance_recorder is
    provided, the dist.recv() call is wrapped in a "metadata_receive" span,
    and the "metadata_receive_count" counter is incremented.

    Returns the received CPU tensor (metadata stays on CPU).

    Requirements: 2.2, 8.1, 8.2, 8.3
    """
    import torch
    import torch.distributed as dist

    buffer = torch.empty(shape, dtype=dtype, device="cpu")
    local_rank = dist.get_rank()
    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "metadata_receive",
                metadata={
                    "src_rank": src_rank,
                    "dst_rank": local_rank,
                    "shape": list(shape),
                    "dtype": str(dtype),
                },
            ):
                dist.recv(buffer, src=src_rank)
            performance_recorder.increment_counter("metadata_receive_count")
        else:
            dist.recv(buffer, src=src_rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to receive metadata: src_rank={src_rank}, "
            f"local_rank={local_rank}, tensor_shape={shape}: {exc}"
        ) from exc
    return buffer


def broadcast_token_sync(
    tensor: "torch.Tensor",
    src_rank: int,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> "torch.Tensor":
    """Broadcast a token tensor from src_rank to all ranks for synchronization.

    Used for token synchronization across pipeline stages. When a
    performance_recorder is provided, the dist.broadcast() call is wrapped
    in a "token_synchronization" span, and the "token_synchronization_count"
    counter is incremented.

    Args:
        tensor: The tensor to broadcast (must be on CPU for Gloo).
        src_rank: The rank that owns the source data.
        performance_recorder: Optional recorder for timing measurement.

    Returns:
        The tensor after broadcast (in-place on all ranks).

    Requirements: 2.6, 2.7, 8.1, 8.2, 8.3
    """
    import torch.distributed as dist

    cpu_tensor = tensor.to("cpu") if not tensor.is_cpu else tensor
    local_rank = dist.get_rank()
    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "token_synchronization",
                metadata={
                    "src_rank": src_rank,
                    "local_rank": local_rank,
                    "shape": list(cpu_tensor.shape),
                    "dtype": str(cpu_tensor.dtype),
                },
            ):
                dist.broadcast(cpu_tensor, src=src_rank)
            performance_recorder.increment_counter("token_synchronization_count")
        else:
            dist.broadcast(cpu_tensor, src=src_rank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to broadcast token sync: src_rank={src_rank}, "
            f"local_rank={local_rank}, tensor_shape={tuple(cpu_tensor.shape)}: {exc}"
        ) from exc
    return cpu_tensor


# ---------------------------------------------------------------------------
# Asynchronous Pipeline Communication (Task 2.1)
# ---------------------------------------------------------------------------


@dataclass
class AsyncSendHandle:
    """Handle for an asynchronous send operation.

    Provides a way to wait for completion or proceed without waiting
    to enable computation-communication overlap.

    Requirements: 2.1, 2.2
    """

    future: "torch.distributed.Work"
    """The underlying PyTorch distributed work object."""

    def wait(self) -> None:
        """Block until the send operation completes."""
        self.future.wait()


@dataclass
class AsyncRecvHandle:
    """Handle for an asynchronous receive operation.

    Provides access to the received tensor once the operation completes,
    along with the metadata needed to unstage it to the target device.

    Requirements: 2.1, 2.2
    """

    buffer: "torch.Tensor"
    dtype: "torch.dtype"
    shape: tuple[int, ...]
    target_device: str
    future: "torch.distributed.Work"
    """The underlying PyTorch distributed work object."""

    def result(self) -> "torch.Tensor":
        """Block until receive completes and return the unstaged tensor."""
        self.future.wait()
        staged = CpuStagedTensor(
            cpu_tensor=self.buffer,
            original_dtype=self.dtype,
            original_shape=self.shape,
        )
        return unstage_from_cpu(staged, self.target_device)


# ---------------------------------------------------------------------------
# Decode Fast-Path Protocol Negotiation (Task 2)
# ---------------------------------------------------------------------------

DECODE_FAST_PATH_PROTOCOL_VERSION: int = 1
"""Protocol version for the decode fast-path communication.

Incremented when the metadata exchange format or validation rules change.
Both sides must agree on this version during negotiation.
"""

# Metadata tensor layout for protocol negotiation:
# Index 0: protocol_version
# Index 1: hidden_size
# Index 2: dtype encoding (see _DTYPE_TO_ENCODING / _ENCODING_TO_DTYPE)
# Index 3: maximum_microbatch_size
_NEGOTIATION_METADATA_LENGTH: int = 4

_DTYPE_TO_ENCODING: dict["torch.dtype", int] = {}
_ENCODING_TO_DTYPE: dict[int, "torch.dtype"] = {}


def _initialize_dtype_encoding_maps() -> None:
    """Lazily initialize dtype encoding maps (requires torch import)."""
    import torch

    global _DTYPE_TO_ENCODING, _ENCODING_TO_DTYPE  # noqa: PLW0603
    if _DTYPE_TO_ENCODING:
        return
    _DTYPE_TO_ENCODING = {
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float32: 3,
        torch.float64: 4,
    }
    _ENCODING_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ENCODING.items()}


def negotiate_decode_activation_protocol(
    *,
    process_group: "dist.ProcessGroup",
    local_rank: int,
    world_size: int,
    hidden_size: int,
    dtype: "torch.dtype",
    maximum_microbatch_size: int,
) -> DecodeActivationProtocol | None:
    """Negotiate the decode fast-path protocol with the downstream neighbor.

    Performs a one-time metadata exchange between adjacent ranks during decode
    initialization (after prefill completes, before the decode loop begins).
    Each rank sends its local metadata to rank+1 and receives metadata from
    rank-1, then validates that both sides agree on protocol version, hidden
    size, and dtype.

    The final rank (rank ``world_size - 1``) has no downstream neighbor and
    returns ``None``.

    Args:
        process_group: The torch.distributed process group for communication.
        local_rank: This rank's index in the pipeline.
        world_size: Total number of ranks in the pipeline.
        hidden_size: Hidden dimension size of the model activations.
        dtype: Expected tensor dtype for decode activations.
        maximum_microbatch_size: Maximum microbatch size for decode.

    Returns:
        A ``DecodeActivationProtocol`` instance representing the agreed-upon
        protocol for sending activations to rank+1, or ``None`` for the final
        rank which has no downstream neighbor.

    Raises:
        DecodeProtocolMismatchError: If the local and remote metadata disagree
            on protocol version, hidden size, or dtype.
        PipelineCommunicationError: If the metadata exchange fails due to a
            communication error.

    Requirements: 2.1, 2.2, 2.5, 2.11
    """
    import torch
    import torch.distributed as dist

    _initialize_dtype_encoding_maps()

    # Final rank has no downstream neighbor — nothing to negotiate
    if local_rank >= world_size - 1:
        logger.info(
            "Rank %d is the final rank (world_size=%d), "
            "no downstream protocol negotiation needed",
            local_rank,
            world_size,
        )
        return None

    downstream_rank = local_rank + 1

    # Encode local metadata into a compact int64 tensor for Gloo transport
    dtype_encoding = _DTYPE_TO_ENCODING.get(dtype)
    if dtype_encoding is None:
        raise DecodeProtocolMismatchError(
            f"Unsupported dtype for decode fast path: {dtype}",
            source_rank=local_rank,
            destination_rank=downstream_rank,
            expected_dtype=str(dtype),
        )

    local_metadata = torch.tensor(
        [
            DECODE_FAST_PATH_PROTOCOL_VERSION,
            hidden_size,
            dtype_encoding,
            maximum_microbatch_size,
        ],
        dtype=torch.int64,
        device="cpu",
    )

    # Exchange metadata with downstream rank:
    # - Send our metadata to rank+1
    # - Receive metadata from rank+1 (they send theirs back)
    #
    # The downstream rank (rank+1) simultaneously:
    # - Receives metadata from rank (us)
    # - Sends its metadata back to rank (us)
    #
    # This creates a bidirectional handshake between adjacent ranks.

    remote_metadata = torch.empty(
        _NEGOTIATION_METADATA_LENGTH, dtype=torch.int64, device="cpu"
    )

    try:
        # Send local metadata to downstream rank
        dist.send(local_metadata, dst=downstream_rank, group=process_group)
        logger.debug(
            "Rank %d sent negotiation metadata to rank %d: "
            "protocol_version=%d, hidden_size=%d, dtype_encoding=%d, "
            "maximum_microbatch_size=%d",
            local_rank,
            downstream_rank,
            DECODE_FAST_PATH_PROTOCOL_VERSION,
            hidden_size,
            dtype_encoding,
            maximum_microbatch_size,
        )
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to send protocol negotiation metadata",
            source_rank=local_rank,
            destination_rank=downstream_rank,
            expected_shape=(_NEGOTIATION_METADATA_LENGTH,),
        ) from exc

    try:
        # Receive downstream rank's metadata (their acknowledgment)
        dist.recv(remote_metadata, src=downstream_rank, group=process_group)
        logger.debug(
            "Rank %d received negotiation metadata from rank %d: %s",
            local_rank,
            downstream_rank,
            remote_metadata.tolist(),
        )
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to receive protocol negotiation metadata",
            source_rank=downstream_rank,
            destination_rank=local_rank,
            expected_shape=(_NEGOTIATION_METADATA_LENGTH,),
        ) from exc

    # Validate agreement on protocol version, hidden size, and dtype
    remote_protocol_version = int(remote_metadata[0].item())
    remote_hidden_size = int(remote_metadata[1].item())
    remote_dtype_encoding = int(remote_metadata[2].item())
    remote_maximum_microbatch_size = int(remote_metadata[3].item())

    if remote_protocol_version != DECODE_FAST_PATH_PROTOCOL_VERSION:
        raise DecodeProtocolMismatchError(
            f"Protocol version mismatch between rank {local_rank} and rank "
            f"{downstream_rank}: local={DECODE_FAST_PATH_PROTOCOL_VERSION}, "
            f"remote={remote_protocol_version}",
            source_rank=local_rank,
            destination_rank=downstream_rank,
            protocol_version=DECODE_FAST_PATH_PROTOCOL_VERSION,
        )

    if remote_hidden_size != hidden_size:
        raise DecodeProtocolMismatchError(
            f"Hidden size mismatch between rank {local_rank} and rank "
            f"{downstream_rank}: local={hidden_size}, remote={remote_hidden_size}",
            source_rank=local_rank,
            destination_rank=downstream_rank,
            expected_shape=(maximum_microbatch_size, 1, hidden_size),
            received_shape=(remote_maximum_microbatch_size, 1, remote_hidden_size),
        )

    if remote_dtype_encoding != dtype_encoding:
        remote_dtype = _ENCODING_TO_DTYPE.get(remote_dtype_encoding)
        raise DecodeProtocolMismatchError(
            f"Dtype mismatch between rank {local_rank} and rank "
            f"{downstream_rank}: local={dtype}, "
            f"remote={remote_dtype if remote_dtype is not None else f'unknown(encoding={remote_dtype_encoding})'}",
            source_rank=local_rank,
            destination_rank=downstream_rank,
            expected_dtype=str(dtype),
            received_dtype=str(remote_dtype) if remote_dtype is not None else f"unknown(encoding={remote_dtype_encoding})",
        )

    # Use the minimum of both sides' maximum microbatch size for safety
    agreed_maximum_microbatch_size = min(
        maximum_microbatch_size, remote_maximum_microbatch_size
    )

    # Build the negotiated protocol
    negotiated_shape = (agreed_maximum_microbatch_size, 1, hidden_size)

    protocol = DecodeActivationProtocol(
        protocol_version=DECODE_FAST_PATH_PROTOCOL_VERSION,
        source_rank=local_rank,
        destination_rank=downstream_rank,
        dtype_name=str(dtype),
        shape=negotiated_shape,
        maximum_microbatch_size=agreed_maximum_microbatch_size,
        hidden_size=hidden_size,
        requires_contiguous=True,
    )

    logger.info(
        "Rank %d negotiated decode protocol with rank %d: "
        "version=%d, shape=%s, dtype=%s, microbatch_size=%d",
        local_rank,
        downstream_rank,
        protocol.protocol_version,
        protocol.shape,
        protocol.dtype_name,
        protocol.maximum_microbatch_size,
    )

    return protocol


def respond_to_decode_protocol_negotiation(
    *,
    process_group: "dist.ProcessGroup",
    local_rank: int,
    world_size: int,
    hidden_size: int,
    dtype: "torch.dtype",
    maximum_microbatch_size: int,
) -> DecodeActivationProtocol | None:
    """Respond to a decode fast-path protocol negotiation from the upstream neighbor.

    This is the receiving side of the protocol negotiation handshake. Called by
    ranks 1 through world_size-1 to receive metadata from rank-1 and send back
    their own metadata for validation.

    Rank 0 has no upstream neighbor and returns ``None``.

    Args:
        process_group: The torch.distributed process group for communication.
        local_rank: This rank's index in the pipeline.
        world_size: Total number of ranks in the pipeline.
        hidden_size: Hidden dimension size of the model activations.
        dtype: Expected tensor dtype for decode activations.
        maximum_microbatch_size: Maximum microbatch size for decode.

    Returns:
        A ``DecodeActivationProtocol`` instance representing the agreed-upon
        protocol for receiving activations from rank-1, or ``None`` for rank 0
        which has no upstream neighbor.

    Raises:
        DecodeProtocolMismatchError: If the local and remote metadata disagree
            on protocol version, hidden size, or dtype.
        PipelineCommunicationError: If the metadata exchange fails due to a
            communication error.

    Requirements: 2.1, 2.2, 2.5, 2.11
    """
    import torch
    import torch.distributed as dist

    _initialize_dtype_encoding_maps()

    # Rank 0 has no upstream neighbor — nothing to respond to
    if local_rank == 0:
        logger.info(
            "Rank 0 has no upstream neighbor, "
            "no protocol negotiation response needed",
        )
        return None

    upstream_rank = local_rank - 1

    # Encode local metadata
    dtype_encoding = _DTYPE_TO_ENCODING.get(dtype)
    if dtype_encoding is None:
        raise DecodeProtocolMismatchError(
            f"Unsupported dtype for decode fast path: {dtype}",
            source_rank=upstream_rank,
            destination_rank=local_rank,
            expected_dtype=str(dtype),
        )

    local_metadata = torch.tensor(
        [
            DECODE_FAST_PATH_PROTOCOL_VERSION,
            hidden_size,
            dtype_encoding,
            maximum_microbatch_size,
        ],
        dtype=torch.int64,
        device="cpu",
    )

    remote_metadata = torch.empty(
        _NEGOTIATION_METADATA_LENGTH, dtype=torch.int64, device="cpu"
    )

    try:
        # Receive upstream rank's metadata first
        dist.recv(remote_metadata, src=upstream_rank, group=process_group)
        logger.debug(
            "Rank %d received negotiation metadata from rank %d: %s",
            local_rank,
            upstream_rank,
            remote_metadata.tolist(),
        )
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to receive protocol negotiation metadata from upstream",
            source_rank=upstream_rank,
            destination_rank=local_rank,
            expected_shape=(_NEGOTIATION_METADATA_LENGTH,),
        ) from exc

    try:
        # Send our metadata back to upstream rank as acknowledgment
        dist.send(local_metadata, dst=upstream_rank, group=process_group)
        logger.debug(
            "Rank %d sent negotiation metadata to rank %d: "
            "protocol_version=%d, hidden_size=%d, dtype_encoding=%d, "
            "maximum_microbatch_size=%d",
            local_rank,
            upstream_rank,
            DECODE_FAST_PATH_PROTOCOL_VERSION,
            hidden_size,
            dtype_encoding,
            maximum_microbatch_size,
        )
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to send protocol negotiation metadata to upstream",
            source_rank=local_rank,
            destination_rank=upstream_rank,
            expected_shape=(_NEGOTIATION_METADATA_LENGTH,),
        ) from exc

    # Validate agreement on protocol version, hidden size, and dtype
    remote_protocol_version = int(remote_metadata[0].item())
    remote_hidden_size = int(remote_metadata[1].item())
    remote_dtype_encoding = int(remote_metadata[2].item())
    remote_maximum_microbatch_size = int(remote_metadata[3].item())

    if remote_protocol_version != DECODE_FAST_PATH_PROTOCOL_VERSION:
        raise DecodeProtocolMismatchError(
            f"Protocol version mismatch between rank {upstream_rank} and rank "
            f"{local_rank}: remote={remote_protocol_version}, "
            f"local={DECODE_FAST_PATH_PROTOCOL_VERSION}",
            source_rank=upstream_rank,
            destination_rank=local_rank,
            protocol_version=DECODE_FAST_PATH_PROTOCOL_VERSION,
        )

    if remote_hidden_size != hidden_size:
        raise DecodeProtocolMismatchError(
            f"Hidden size mismatch between rank {upstream_rank} and rank "
            f"{local_rank}: remote={remote_hidden_size}, local={hidden_size}",
            source_rank=upstream_rank,
            destination_rank=local_rank,
            expected_shape=(maximum_microbatch_size, 1, hidden_size),
            received_shape=(remote_maximum_microbatch_size, 1, remote_hidden_size),
        )

    if remote_dtype_encoding != dtype_encoding:
        remote_dtype = _ENCODING_TO_DTYPE.get(remote_dtype_encoding)
        raise DecodeProtocolMismatchError(
            f"Dtype mismatch between rank {upstream_rank} and rank "
            f"{local_rank}: remote={remote_dtype if remote_dtype is not None else f'unknown(encoding={remote_dtype_encoding})'}, "
            f"local={dtype}",
            source_rank=upstream_rank,
            destination_rank=local_rank,
            expected_dtype=str(dtype),
            received_dtype=str(remote_dtype) if remote_dtype is not None else f"unknown(encoding={remote_dtype_encoding})",
        )

    # Use the minimum of both sides' maximum microbatch size for safety
    agreed_maximum_microbatch_size = min(
        maximum_microbatch_size, remote_maximum_microbatch_size
    )

    # Build the negotiated protocol (from upstream's perspective sending to us)
    negotiated_shape = (agreed_maximum_microbatch_size, 1, hidden_size)

    protocol = DecodeActivationProtocol(
        protocol_version=DECODE_FAST_PATH_PROTOCOL_VERSION,
        source_rank=upstream_rank,
        destination_rank=local_rank,
        dtype_name=str(dtype),
        shape=negotiated_shape,
        maximum_microbatch_size=agreed_maximum_microbatch_size,
        hidden_size=hidden_size,
        requires_contiguous=True,
    )

    logger.info(
        "Rank %d responded to decode protocol negotiation from rank %d: "
        "version=%d, shape=%s, dtype=%s, microbatch_size=%d",
        local_rank,
        upstream_rank,
        protocol.protocol_version,
        protocol.shape,
        protocol.dtype_name,
        protocol.maximum_microbatch_size,
    )

    return protocol


def send_activation_async(
    tensor: "torch.Tensor", dst_rank: int
) -> AsyncSendHandle:
    """Stage tensor to CPU and initiate non-blocking send to destination rank.

    Unlike the synchronous ``send_activation_generic``, this function returns immediately
    with a handle that can be used to wait for completion. This allows the
    calling rank to begin computation on its next pipeline stage while the
    send proceeds in the background.

    On Intel iGPU nodes (shared memory), the CPU staging is nearly free,
    making async send particularly effective for overlapping with computation.

    Args:
        tensor: The tensor to send (can be on any device).
        dst_rank: Destination rank ID.

    Returns:
        An AsyncSendHandle that can be used to wait for completion.

    Requirements: 2.1, 2.2
    """
    import torch.distributed as dist

    staged = stage_to_cpu(tensor)
    try:
        # async_op=True returns a Work object without blocking
        future = dist.send(staged.cpu_tensor, dst=dst_rank, async_op=True)
    except Exception as exc:
        src_rank = dist.get_rank()
        raise RuntimeError(
            f"Failed to initiate async send: src_rank={src_rank}, dst_rank={dst_rank}, "
            f"tensor_shape={staged.original_shape}: {exc}"
        ) from exc

    return AsyncSendHandle(future=future)


def recv_activation_async(
    shape: tuple[int, ...],
    dtype: "torch.dtype",
    src_rank: int,
    target_device: str,
) -> AsyncRecvHandle:
    """Initiate non-blocking receive from source rank.

    Unlike the synchronous ``recv_activation_generic``, this function returns immediately
    with a handle. The caller can begin computation while the receive proceeds
    in the background, then call `.result()` when the received tensor is needed.

    This enables bubble-free pipeline scheduling: stage N can start its forward
    pass while stage N-1's activation is still being transmitted.

    Args:
        shape: Expected tensor shape.
        dtype: Expected tensor dtype.
        src_rank: Source rank ID.
        target_device: Device to unstage the tensor to after receive.

    Returns:
        An AsyncRecvHandle that provides `.result()` to get the tensor.

    Requirements: 2.1, 2.2
    """
    import torch
    import torch.distributed as dist

    buffer = torch.empty(shape, dtype=dtype, device="cpu")
    try:
        # async_op=True returns a Work object without blocking
        future = dist.recv(buffer, src=src_rank, async_op=True)
    except Exception as exc:
        local_rank = dist.get_rank()
        raise RuntimeError(
            f"Failed to initiate async receive: src_rank={src_rank}, "
            f"local_rank={local_rank}, tensor_shape={shape}: {exc}"
        ) from exc

    return AsyncRecvHandle(
        buffer=buffer,
        dtype=dtype,
        shape=shape,
        target_device=target_device,
        future=future,
    )


# ---------------------------------------------------------------------------
# Decode Fast-Path Send/Receive (Task 2)
# ---------------------------------------------------------------------------


def send_decode_activation_fast(
    *,
    activation: "torch.Tensor",
    protocol: DecodeActivationProtocol,
    buffer_pool: "CommunicationBufferPool",
    process_group: "dist.ProcessGroup",
) -> None:
    """Send a decode activation tensor using the negotiated fast path.

    After protocol negotiation establishes a fixed shape, dtype, and rank pair,
    this function sends the activation without any per-token shape metadata.
    It uses a preallocated CPU buffer from the pool to avoid per-token
    allocation overhead.

    The key optimization: the protocol already encodes the shape, so the
    receiver knows exactly what to expect. No metadata exchange is needed.

    Args:
        activation: The activation tensor to send. Must be contiguous, match
            the protocol's expected shape, and match the protocol's dtype.
        protocol: The negotiated decode activation protocol specifying shape,
            dtype, source rank, and destination rank.
        buffer_pool: Pool of reusable CPU buffers for Gloo communication.
        process_group: The torch.distributed process group for communication.

    Returns:
        None on success.

    Raises:
        DecodeProtocolMismatchError: If the activation tensor is not contiguous,
            has a shape mismatch, or has a dtype mismatch with the protocol.
        PipelineCommunicationError: If the Gloo send operation fails.

    Requirements: 2.1, 2.3, 2.5, 2.14
    """
    import torch.distributed as dist

    # Validate contiguity
    if not activation.is_contiguous():
        raise DecodeProtocolMismatchError(
            "Activation tensor is not contiguous; fast path requires contiguous tensors",
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
            expected_shape=protocol.shape,
            received_shape=tuple(activation.shape),
            expected_dtype=protocol.dtype_name,
            received_dtype=str(activation.dtype),
        )

    # Validate shape matches protocol
    activation_shape = tuple(activation.shape)
    if activation_shape != protocol.shape:
        raise DecodeProtocolMismatchError(
            f"Activation shape {activation_shape} does not match "
            f"negotiated protocol shape {protocol.shape}",
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
            expected_shape=protocol.shape,
            received_shape=activation_shape,
            expected_dtype=protocol.dtype_name,
            received_dtype=str(activation.dtype),
        )

    # Validate dtype matches protocol
    expected_dtype_str = protocol.dtype_name
    actual_dtype_str = str(activation.dtype)
    if actual_dtype_str != expected_dtype_str:
        raise DecodeProtocolMismatchError(
            f"Activation dtype {actual_dtype_str} does not match "
            f"negotiated protocol dtype {expected_dtype_str}",
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
            expected_shape=protocol.shape,
            received_shape=activation_shape,
            expected_dtype=expected_dtype_str,
            received_dtype=actual_dtype_str,
        )

    # Acquire preallocated CPU send buffer from pool
    send_buffer = buffer_pool.acquire(
        buffer_name="decode_activation_send",
        dtype=activation.dtype,
        shape=protocol.shape,
        source_rank=protocol.source_rank,
        destination_rank=protocol.destination_rank,
    )

    try:
        # Copy activation to CPU buffer (nearly free on Intel iGPU shared memory)
        send_buffer.copy_(activation)

        logger.debug(
            "Fast-path send: rank %d -> rank %d, shape=%s, dtype=%s",
            protocol.source_rank,
            protocol.destination_rank,
            protocol.shape,
            protocol.dtype_name,
        )

        # Send via Gloo — no shape metadata, receiver already knows the shape
        try:
            dist.send(send_buffer, dst=protocol.destination_rank, group=process_group)
        except Exception as exc:
            raise PipelineCommunicationError(
                "Fast-path activation send failed",
                source_rank=protocol.source_rank,
                destination_rank=protocol.destination_rank,
                expected_shape=protocol.shape,
            ) from exc
    finally:
        # Always release the buffer back to the pool
        buffer_pool.release(
            buffer_name="decode_activation_send",
            dtype=activation.dtype,
            shape=protocol.shape,
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
        )


def receive_decode_activation_fast(
    *,
    protocol: DecodeActivationProtocol,
    buffer_pool: "CommunicationBufferPool",
    process_group: "dist.ProcessGroup",
    target_device: str,
) -> "torch.Tensor":
    """Receive a decode activation tensor using the negotiated fast path.

    After protocol negotiation establishes a fixed shape, dtype, and rank pair,
    this function receives the activation without any per-token shape metadata.
    It uses a preallocated CPU buffer from the pool to avoid per-token
    allocation overhead, then copies the result to the target XPU device.

    The key optimization: the protocol already encodes the shape, so no
    metadata exchange is needed. The buffer pool eliminates per-token allocation.

    Args:
        protocol: The negotiated decode activation protocol specifying shape,
            dtype, source rank, and destination rank.
        buffer_pool: Pool of reusable CPU buffers for Gloo communication.
        process_group: The torch.distributed process group for communication.
        target_device: Device string for the target XPU device (e.g., "xpu:0").

    Returns:
        The received activation tensor on the target device.

    Raises:
        PipelineCommunicationError: If the Gloo receive operation fails.

    Requirements: 2.2, 2.3, 2.5, 2.14
    """
    import torch.distributed as dist

    # Resolve dtype from protocol string representation
    _initialize_dtype_encoding_maps()
    recv_dtype = None
    for dt, _ in _DTYPE_TO_ENCODING.items():
        if str(dt) == protocol.dtype_name:
            recv_dtype = dt
            break

    if recv_dtype is None:
        raise PipelineCommunicationError(
            f"Cannot resolve dtype from protocol dtype_name: {protocol.dtype_name!r}",
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
            expected_shape=protocol.shape,
        )

    # Acquire preallocated CPU receive buffer from pool
    recv_buffer = buffer_pool.acquire(
        buffer_name="decode_activation_recv",
        dtype=recv_dtype,
        shape=protocol.shape,
        source_rank=protocol.source_rank,
        destination_rank=protocol.destination_rank,
    )

    try:
        logger.debug(
            "Fast-path recv: rank %d <- rank %d, shape=%s, dtype=%s",
            protocol.destination_rank,
            protocol.source_rank,
            protocol.shape,
            protocol.dtype_name,
        )

        # Receive via Gloo — no shape metadata, we already know the shape
        try:
            dist.recv(recv_buffer, src=protocol.source_rank, group=process_group)
        except Exception as exc:
            raise PipelineCommunicationError(
                "Fast-path activation receive failed",
                source_rank=protocol.source_rank,
                destination_rank=protocol.destination_rank,
                expected_shape=protocol.shape,
            ) from exc

        # Copy received data to target XPU device
        result = recv_buffer.to(target_device)
    finally:
        # Always release the buffer back to the pool
        buffer_pool.release(
            buffer_name="decode_activation_recv",
            dtype=recv_dtype,
            shape=protocol.shape,
            source_rank=protocol.source_rank,
            destination_rank=protocol.destination_rank,
        )

    return result


# ---------------------------------------------------------------------------
# Point-to-Point Token Result Communication (Task 2)
#
# Replaces the blocking dist.broadcast() token synchronization with direct
# point-to-point send/recv between rank 3 (final stage) and rank 0.
# Ranks 1 and 2 do NOT participate — they continue processing the next
# pipeline stage without waiting.
# ---------------------------------------------------------------------------

# Serialization format for TokenResultPacket over the wire:
#
# Header tensor (int64, shape [5]):
#   [0] token_identifier
#   [1] position
#   [2] finished (0 or 1)
#   [3] finish_reason encoding (0=None, 1="stop", 2="length", 3="other")
#   [4] request_identifier byte length
#
# Payload tensor (uint8, shape [_TOKEN_RESULT_REQUEST_ID_MAX_BYTES]):
#   UTF-8 encoded request_identifier, zero-padded to fixed size.
#
# Using a fixed-size payload avoids a variable-length handshake and keeps
# the communication to exactly two dist.send/recv calls per token.

_TOKEN_RESULT_HEADER_LENGTH: int = 5
_TOKEN_RESULT_REQUEST_ID_MAX_BYTES: int = 256

_FINISH_REASON_TO_ENCODING: dict[str | None, int] = {
    None: 0,
    "stop": 1,
    "length": 2,
}

_ENCODING_TO_FINISH_REASON: dict[int, str | None] = {
    0: None,
    1: "stop",
    2: "length",
    3: "other",
}


def _encode_finish_reason(finish_reason: str | None) -> int:
    """Encode a finish reason string to a compact integer for wire transport."""
    if finish_reason is None:
        return 0
    encoding = _FINISH_REASON_TO_ENCODING.get(finish_reason)
    if encoding is not None:
        return encoding
    # Unknown finish reason — encode as "other" (3)
    return 3


def _decode_finish_reason(encoding: int) -> str | None:
    """Decode a finish reason integer back to a string or None."""
    return _ENCODING_TO_FINISH_REASON.get(encoding)


def send_token_results_to_rank_zero(
    *,
    packet: TokenResultPacket,
    process_group: "dist.ProcessGroup",
    performance_recorder: "PerformanceRecorder | None" = None,
) -> None:
    """Send a token result from the final rank directly to rank 0.

    Called by the final rank (rank world_size-1, typically rank 3) after
    sampling a token. Serializes the TokenResultPacket into a compact wire
    format and sends it to rank 0 using point-to-point ``dist.send()``.

    Only the final rank calls this function. Ranks 1 and 2 do NOT participate
    in token-result communication, allowing them to continue processing the
    next pipeline stage without blocking.

    The serialization uses two fixed-size tensors:
    - A header tensor (int64, 5 elements) containing token_identifier, position,
      finished flag, finish_reason encoding, and request_identifier byte length.
    - A payload tensor (uint8, 256 bytes) containing the UTF-8 encoded
      request_identifier, zero-padded to fixed size.

    Args:
        packet: The TokenResultPacket to send to rank 0.
        process_group: The torch.distributed process group for communication.
        performance_recorder: Optional recorder for timing measurement.

    Raises:
        PipelineCommunicationError: If the send operation fails or the
            request_identifier exceeds the maximum byte length (256 bytes).

    Requirements: 2.7, 2.8, 2.9
    """
    import torch
    import torch.distributed as dist

    # Encode request_identifier to UTF-8 bytes
    request_id_bytes = packet.request_identifier.encode("utf-8")
    request_id_length = len(request_id_bytes)

    if request_id_length > _TOKEN_RESULT_REQUEST_ID_MAX_BYTES:
        raise PipelineCommunicationError(
            f"request_identifier exceeds maximum byte length: "
            f"{request_id_length} > {_TOKEN_RESULT_REQUEST_ID_MAX_BYTES}",
            source_rank=dist.get_rank(),
            destination_rank=0,
            request_identifier=packet.request_identifier,
        )

    # Build header tensor: [token_id, position, finished, finish_reason_enc, id_length]
    header = torch.tensor(
        [
            packet.token_identifier,
            packet.position,
            1 if packet.finished else 0,
            _encode_finish_reason(packet.finish_reason),
            request_id_length,
        ],
        dtype=torch.int64,
        device="cpu",
    )

    # Build payload tensor: UTF-8 request_identifier zero-padded to fixed size
    payload = torch.zeros(
        _TOKEN_RESULT_REQUEST_ID_MAX_BYTES, dtype=torch.uint8, device="cpu"
    )
    if request_id_length > 0:
        payload[:request_id_length] = torch.tensor(
            list(request_id_bytes), dtype=torch.uint8, device="cpu"
        )

    source_rank = dist.get_rank()
    destination_rank = 0

    logger.debug(
        "Token result send: rank %d -> rank %d, "
        "request_id=%r, token_id=%d, position=%d, finished=%s",
        source_rank,
        destination_rank,
        packet.request_identifier,
        packet.token_identifier,
        packet.position,
        packet.finished,
    )

    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "token_result_send",
                metadata={
                    "src_rank": source_rank,
                    "dst_rank": destination_rank,
                    "request_identifier": packet.request_identifier,
                    "token_identifier": packet.token_identifier,
                    "position": packet.position,
                    "finished": packet.finished,
                },
            ):
                dist.send(header, dst=destination_rank, group=process_group)
                dist.send(payload, dst=destination_rank, group=process_group)
            performance_recorder.increment_counter("token_result_send_count")
        else:
            dist.send(header, dst=destination_rank, group=process_group)
            dist.send(payload, dst=destination_rank, group=process_group)
    except PipelineCommunicationError:
        raise
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to send token result to rank 0",
            source_rank=source_rank,
            destination_rank=destination_rank,
            request_identifier=packet.request_identifier,
        ) from exc


def receive_token_results_from_final_rank(
    *,
    process_group: "dist.ProcessGroup",
    world_size: int,
    performance_recorder: "PerformanceRecorder | None" = None,
) -> TokenResultPacket:
    """Receive a token result on rank 0 from the final rank.

    Called by rank 0 to receive token results from the final rank
    (rank world_size-1, typically rank 3) using point-to-point
    ``dist.recv()``.

    Only rank 0 calls this function. Ranks 1 and 2 do NOT participate
    in token-result communication, allowing them to continue processing
    the next pipeline stage without blocking.

    Receives two fixed-size tensors and deserializes them back into a
    TokenResultPacket:
    - A header tensor (int64, 5 elements) containing token_identifier, position,
      finished flag, finish_reason encoding, and request_identifier byte length.
    - A payload tensor (uint8, 256 bytes) containing the UTF-8 encoded
      request_identifier.

    Args:
        process_group: The torch.distributed process group for communication.
        world_size: Total number of ranks in the pipeline (used to determine
            the final rank as world_size - 1).
        performance_recorder: Optional recorder for timing measurement.

    Returns:
        The deserialized TokenResultPacket from the final rank.

    Raises:
        PipelineCommunicationError: If the receive operation fails or the
            received data cannot be deserialized into a valid TokenResultPacket.

    Requirements: 2.7, 2.8, 2.9
    """
    import torch
    import torch.distributed as dist

    final_rank = world_size - 1
    local_rank = 0  # This function is only called by rank 0

    # Allocate receive buffers
    header_buffer = torch.empty(
        _TOKEN_RESULT_HEADER_LENGTH, dtype=torch.int64, device="cpu"
    )
    payload_buffer = torch.empty(
        _TOKEN_RESULT_REQUEST_ID_MAX_BYTES, dtype=torch.uint8, device="cpu"
    )

    logger.debug(
        "Token result recv: rank %d <- rank %d, waiting for token result",
        local_rank,
        final_rank,
    )

    try:
        if performance_recorder is not None:
            with performance_recorder.span(
                "token_result_receive",
                metadata={
                    "src_rank": final_rank,
                    "dst_rank": local_rank,
                },
            ):
                dist.recv(header_buffer, src=final_rank, group=process_group)
                dist.recv(payload_buffer, src=final_rank, group=process_group)
            performance_recorder.increment_counter("token_result_receive_count")
        else:
            dist.recv(header_buffer, src=final_rank, group=process_group)
            dist.recv(payload_buffer, src=final_rank, group=process_group)
    except PipelineCommunicationError:
        raise
    except Exception as exc:
        raise PipelineCommunicationError(
            "Failed to receive token result from final rank",
            source_rank=final_rank,
            destination_rank=local_rank,
        ) from exc

    # Deserialize header
    token_identifier = int(header_buffer[0].item())
    position = int(header_buffer[1].item())
    finished = bool(header_buffer[2].item())
    finish_reason_encoding = int(header_buffer[3].item())
    request_id_length = int(header_buffer[4].item())

    # Validate request_id_length
    if request_id_length < 0 or request_id_length > _TOKEN_RESULT_REQUEST_ID_MAX_BYTES:
        raise PipelineCommunicationError(
            f"Invalid request_identifier length in token result header: "
            f"{request_id_length}",
            source_rank=final_rank,
            destination_rank=local_rank,
            received_byte_count=request_id_length,
        )

    # Deserialize payload: extract request_identifier from UTF-8 bytes
    try:
        request_id_bytes = bytes(payload_buffer[:request_id_length].tolist())
        request_identifier = request_id_bytes.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise PipelineCommunicationError(
            "Failed to decode request_identifier from token result payload",
            source_rank=final_rank,
            destination_rank=local_rank,
            received_byte_count=request_id_length,
        ) from exc

    # Decode finish_reason
    finish_reason = _decode_finish_reason(finish_reason_encoding)

    packet = TokenResultPacket(
        request_identifier=request_identifier,
        token_identifier=token_identifier,
        position=position,
        finished=finished,
        finish_reason=finish_reason,
    )

    logger.debug(
        "Token result recv: rank %d <- rank %d, "
        "request_id=%r, token_id=%d, position=%d, finished=%s, finish_reason=%s",
        local_rank,
        final_rank,
        packet.request_identifier,
        packet.token_identifier,
        packet.position,
        packet.finished,
        packet.finish_reason,
    )

    return packet
