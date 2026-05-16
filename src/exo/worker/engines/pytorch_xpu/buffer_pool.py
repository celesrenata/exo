"""
Communication buffer pool for pipeline-parallel decode fast path.

Manages reusable CPU-pinned tensors for Gloo send/receive operations during
steady-state decode. Buffers are keyed by name, dtype, shape, source rank, and
destination rank so that the decode loop avoids per-token tensor allocation.

The Gloo backend requires all tensors to reside on CPU for send/recv. On Intel
iGPU nodes with shared memory architecture, the CPU↔GPU copy is nearly free,
but repeated ``torch.empty(...)`` calls inside the decode loop add measurable
overhead from Python object creation, allocator bookkeeping, and memory
fragmentation. This pool eliminates that overhead by allocating each unique
buffer shape exactly once and reusing it across decode steps.

Usage::

    pool = CommunicationBufferPool()

    # Acquire a buffer (allocates on first call, reuses thereafter)
    send_buffer = pool.acquire(
        buffer_name="decode_activation_send",
        dtype=torch.bfloat16,
        shape=(1, 1, 3584),
        source_rank=0,
        destination_rank=1,
    )

    # Use the buffer for Gloo communication...
    send_buffer.copy_(activation_tensor)
    dist.send(send_buffer, dst=1)

    # Release the buffer back to the pool when done
    pool.release(
        buffer_name="decode_activation_send",
        dtype=torch.bfloat16,
        shape=(1, 1, 3584),
        source_rank=0,
        destination_rank=1,
    )

    # Clear all buffers (e.g., on protocol renegotiation)
    pool.clear()

**Validates: Requirements 2.3, 2.4, 2.6**
"""

from __future__ import annotations

import logging
from typing import final

import torch
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Buffer Key — immutable identifier for a pooled communication buffer
# ---------------------------------------------------------------------------


@final
class CommunicationBufferKey(BaseModel):
    """Immutable composite key identifying a unique communication buffer.

    Buffers are distinguished by their logical name (e.g., "decode_activation_send"),
    the tensor dtype and shape, and the source/destination rank pair. This ensures
    that each directional communication channel between two ranks gets its own
    dedicated buffer without collision.

    The model is frozen to allow safe use as a dictionary key (via its hash)
    and to prevent accidental mutation after creation.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    buffer_name: str
    """Logical name identifying the purpose of this buffer (e.g., 'decode_activation_send')."""

    dtype_name: str
    """String representation of the tensor dtype (e.g., 'torch.bfloat16').

    Stored as a string rather than ``torch.dtype`` because Pydantic strict mode
    does not natively validate torch dtype objects.
    """

    shape: tuple[int, ...]
    """Expected tensor shape for this buffer."""

    source_rank: int
    """Rank that sends data through this buffer."""

    destination_rank: int
    """Rank that receives data through this buffer."""


# ---------------------------------------------------------------------------
# Pool Statistics — snapshot of buffer pool state
# ---------------------------------------------------------------------------


@final
class BufferPoolStatistics(BaseModel):
    """Snapshot of the communication buffer pool state.

    Provides visibility into pool utilization for instrumentation and debugging.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    total_buffer_count: int
    """Total number of buffers currently managed by the pool (active + available)."""

    active_buffer_count: int
    """Number of buffers currently acquired and in use."""

    available_buffer_count: int
    """Number of buffers available for reuse (total - active)."""

    reuse_count: int
    """Cumulative count of buffer acquisitions that reused an existing buffer."""

    allocation_count: int
    """Cumulative count of buffer allocations (first-time creations)."""


# ---------------------------------------------------------------------------
# CommunicationBufferPool — reusable CPU tensor pool for Gloo communication
# ---------------------------------------------------------------------------


@final
class CommunicationBufferPool:
    """Pool of reusable CPU tensors for Gloo send/receive operations.

    Eliminates per-token tensor allocation in the decode fast path by maintaining
    a registry of preallocated CPU buffers keyed by their communication identity
    (name, dtype, shape, source rank, destination rank).

    Thread safety: This class is NOT thread-safe. It is designed to be used from
    a single thread per rank in the pipeline-parallel decode loop.

    Lifecycle:
        1. ``acquire(...)`` — returns a buffer for the given key. Allocates on
           first call, reuses the existing buffer on subsequent calls.
        2. ``release(...)`` — marks the buffer as available for reuse. The buffer
           remains allocated and is not freed.
        3. ``clear()`` — deallocates all buffers and resets the pool. Used during
           protocol renegotiation or shutdown.
    """

    def __init__(self) -> None:
        """Initialize an empty buffer pool with zero allocations."""
        self._buffers: dict[CommunicationBufferKey, torch.Tensor] = {}
        self._active_keys: set[CommunicationBufferKey] = set()
        self._reuse_count: int = 0
        self._allocation_count: int = 0

    def acquire(
        self,
        *,
        buffer_name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        source_rank: int,
        destination_rank: int,
    ) -> torch.Tensor:
        """Acquire a CPU buffer for communication, allocating if necessary.

        On first call for a given key, allocates a new CPU tensor with the
        specified dtype and shape. On subsequent calls with the same key,
        returns the previously allocated tensor without new allocation.

        The returned tensor is on CPU device, suitable for Gloo send/recv.
        Its contents are undefined on acquisition — the caller must fill it
        before sending.

        Args:
            buffer_name: Logical name for this buffer (e.g., 'decode_activation_send').
            dtype: PyTorch dtype for the buffer tensor.
            shape: Shape of the buffer tensor.
            source_rank: Rank that sends through this buffer.
            destination_rank: Rank that receives through this buffer.

        Returns:
            A CPU tensor with the specified dtype and shape.

        Raises:
            RuntimeError: If the buffer is already acquired and not yet released.
        """
        key = CommunicationBufferKey(
            buffer_name=buffer_name,
            dtype_name=str(dtype),
            shape=shape,
            source_rank=source_rank,
            destination_rank=destination_rank,
        )

        if key in self._active_keys:
            raise RuntimeError(
                f"Buffer already acquired and not released: "
                f"name={buffer_name!r}, dtype={dtype}, shape={shape}, "
                f"source_rank={source_rank}, destination_rank={destination_rank}"
            )

        if key in self._buffers:
            # Reuse existing buffer
            self._reuse_count += 1
            self._active_keys.add(key)
            logger.debug(
                "Reusing buffer: name=%r, shape=%s, source=%d, dest=%d",
                buffer_name,
                shape,
                source_rank,
                destination_rank,
            )
            return self._buffers[key]

        # Allocate new buffer on CPU for Gloo compatibility
        buffer = torch.empty(shape, dtype=dtype, device="cpu")
        self._buffers[key] = buffer
        self._active_keys.add(key)
        self._allocation_count += 1

        logger.debug(
            "Allocated new buffer: name=%r, dtype=%s, shape=%s, source=%d, dest=%d",
            buffer_name,
            dtype,
            shape,
            source_rank,
            destination_rank,
        )

        return buffer

    def release(
        self,
        *,
        buffer_name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        source_rank: int,
        destination_rank: int,
    ) -> None:
        """Release a buffer back to the pool for future reuse.

        The buffer remains allocated in memory but is marked as available.
        A subsequent ``acquire(...)`` call with the same key will return the
        same tensor without allocation.

        Args:
            buffer_name: Logical name for this buffer.
            dtype: PyTorch dtype of the buffer.
            shape: Shape of the buffer.
            source_rank: Rank that sends through this buffer.
            destination_rank: Rank that receives through this buffer.

        Raises:
            RuntimeError: If the buffer was not previously acquired.
        """
        key = CommunicationBufferKey(
            buffer_name=buffer_name,
            dtype_name=str(dtype),
            shape=shape,
            source_rank=source_rank,
            destination_rank=destination_rank,
        )

        if key not in self._active_keys:
            raise RuntimeError(
                f"Cannot release buffer that is not acquired: "
                f"name={buffer_name!r}, dtype={dtype}, shape={shape}, "
                f"source_rank={source_rank}, destination_rank={destination_rank}"
            )

        self._active_keys.discard(key)

    def clear(self) -> None:
        """Clear all buffers from the pool, releasing memory.

        Resets the pool to its initial empty state. Use this during protocol
        renegotiation, shutdown, or when the decode microbatch layout changes
        and existing buffers are no longer valid.

        Active buffers are forcibly released. The reuse and allocation counters
        are preserved for lifetime statistics.
        """
        buffer_count = len(self._buffers)
        active_count = len(self._active_keys)

        if active_count > 0:
            logger.warning(
                "Clearing buffer pool with %d active buffers still acquired",
                active_count,
            )

        self._buffers.clear()
        self._active_keys.clear()

        logger.debug(
            "Buffer pool cleared: freed %d buffers (%d were active)",
            buffer_count,
            active_count,
        )

    def get_statistics(self) -> BufferPoolStatistics:
        """Return a snapshot of the current pool state.

        Provides visibility into pool utilization for instrumentation,
        debugging, and performance monitoring.

        Returns:
            An immutable statistics snapshot with buffer counts and reuse metrics.
        """
        total = len(self._buffers)
        active = len(self._active_keys)

        return BufferPoolStatistics(
            total_buffer_count=total,
            active_buffer_count=active,
            available_buffer_count=total - active,
            reuse_count=self._reuse_count,
            allocation_count=self._allocation_count,
        )
