"""
Decode output buffer pool for GatedDeltaNet post-processing intermediate tensors.

Manages reusable XPU/CPU tensors for the decode path's post-processing steps
(RMSNorm, gating, output projection) during steady-state decode. Buffers are
keyed by layer index, logical name, shape, dtype, and device so that the decode
loop avoids per-token tensor allocation for intermediate results.

During decode, the microbatch shape is stable (batch_size × 1 × hidden_dim for
single-token generation). This pool exploits that stability by allocating each
unique buffer shape once per layer and reusing it across decode steps. If the
shape changes (e.g., batch size changes during continuous batching), the pool
reallocates the buffer and logs a warning.

Usage::

    pool = DecodeOutputBufferPool()

    # Get a buffer for layer 3's RMSNorm intermediate
    buffer = pool.get_buffer(
        layer_index=3,
        buffer_name="rmsnorm_intermediate",
        shape=(1, 1, 3584),
        dtype=torch.bfloat16,
        device=torch.device("xpu:0"),
    )

    # Use the buffer (write into it)
    torch.mul(input_tensor, norm_weight, out=buffer)

    # On next decode step, same call returns the same buffer (no allocation)
    buffer_again = pool.get_buffer(
        layer_index=3,
        buffer_name="rmsnorm_intermediate",
        shape=(1, 1, 3584),
        dtype=torch.bfloat16,
        device=torch.device("xpu:0"),
    )
    assert buffer.data_ptr() == buffer_again.data_ptr()

    # Clear all buffers on request completion
    pool.clear()

**Validates: Requirements 5.3, 5.4**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import final

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Buffer Key — immutable identifier for a pooled decode output buffer
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class DecodeOutputBufferKey:
    """Immutable composite key identifying a unique decode output buffer.

    Buffers are distinguished by their layer index, logical name (e.g.,
    "rmsnorm_intermediate"), tensor dtype, shape, and device. This ensures
    each layer's post-processing step gets its own dedicated buffer without
    collision.
    """

    layer_index: int
    """Global layer index this buffer belongs to (0-indexed)."""

    buffer_name: str
    """Logical name identifying the purpose of this buffer."""

    shape: tuple[int, ...]
    """Expected tensor shape for this buffer."""

    dtype_name: str
    """String representation of the tensor dtype (e.g., 'torch.bfloat16')."""

    device_name: str
    """String representation of the device (e.g., 'xpu:0', 'cpu')."""


# ---------------------------------------------------------------------------
# Pool Statistics — snapshot of decode output buffer pool state
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class DecodeOutputBufferPoolStatistics:
    """Snapshot of the decode output buffer pool state.

    Provides visibility into pool utilization for instrumentation and debugging.
    """

    total_buffer_count: int
    """Total number of buffers currently managed by the pool."""

    allocation_count: int
    """Cumulative count of buffer allocations (first-time creations)."""

    reuse_count: int
    """Cumulative count of buffer acquisitions that reused an existing buffer."""

    reallocation_count: int
    """Cumulative count of buffer reallocations due to shape changes."""


# ---------------------------------------------------------------------------
# DecodeOutputBufferPool — reusable tensor pool for decode post-processing
# ---------------------------------------------------------------------------


@final
@dataclass(slots=True)
class DecodeOutputBufferPool:
    """Pool of reusable tensors for decode output post-processing intermediates.

    Eliminates per-token tensor allocation in the decode path by maintaining
    a registry of preallocated buffers keyed by layer index, buffer name,
    shape, dtype, and device.

    When a buffer is requested with a shape that differs from the stored buffer
    for the same (layer_index, buffer_name), the pool reallocates the buffer
    and logs a warning. This handles continuous batching scenarios where batch
    size changes.

    Thread safety: This class is NOT thread-safe. It is designed to be used from
    a single thread per rank in the pipeline-parallel decode loop.

    Lifecycle:
        1. ``get_buffer(...)`` — returns a buffer for the given key. Allocates on
           first call, reuses the existing buffer on subsequent calls with matching
           shape. Reallocates on shape change.
        2. ``clear()`` — deallocates all buffers and resets the pool. Used on
           request completion or cancellation.
        3. ``clear_layer(layer_index)`` — deallocates buffers for a specific layer.
    """

    _buffers: dict[DecodeOutputBufferKey, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )
    """Registry of allocated buffers keyed by their identity."""

    _lookup_cache: dict[tuple[int, str], DecodeOutputBufferKey] = field(
        default_factory=dict, init=False, repr=False
    )
    """Fast lookup from (layer_index, buffer_name) to the current key.

    This allows detecting shape changes without scanning all keys.
    """

    _allocation_count: int = field(default=0, init=False, repr=False)
    """Cumulative count of buffer allocations."""

    _reuse_count: int = field(default=0, init=False, repr=False)
    """Cumulative count of buffer reuses."""

    _reallocation_count: int = field(default=0, init=False, repr=False)
    """Cumulative count of buffer reallocations due to shape change."""

    def get_buffer(
        self,
        *,
        layer_index: int,
        buffer_name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get a decode output buffer, allocating or reusing as appropriate.

        On first call for a given (layer_index, buffer_name) combination,
        allocates a new tensor with the specified dtype, shape, and device.
        On subsequent calls with the same shape, returns the previously
        allocated tensor without new allocation.

        If the shape differs from the stored buffer (e.g., batch size changed),
        the old buffer is discarded, a new one is allocated, and a warning is
        logged.

        The returned tensor's contents are undefined — the caller must write
        into it before reading.

        Args:
            layer_index: Global layer index (0-indexed).
            buffer_name: Logical name for this buffer (e.g., 'rmsnorm_intermediate').
            dtype: PyTorch dtype for the buffer tensor.
            shape: Shape of the buffer tensor.
            device: Device for the buffer tensor.

        Returns:
            A tensor with the specified dtype, shape, and device.
        """
        lookup_key = (layer_index, buffer_name)
        dtype_name = str(dtype)
        device_name = str(device)

        # Check if we have an existing buffer for this (layer, name) pair
        existing_key = self._lookup_cache.get(lookup_key)

        if existing_key is not None:
            # Check if shape, dtype, and device still match
            if (
                existing_key.shape == shape
                and existing_key.dtype_name == dtype_name
                and existing_key.device_name == device_name
            ):
                # Reuse existing buffer
                self._reuse_count += 1
                return self._buffers[existing_key]

            # Shape/dtype/device changed — reallocate
            logger.warning(
                "Decode output buffer shape changed: layer=%d, name=%r, "
                "old_shape=%s, new_shape=%s, old_dtype=%s, new_dtype=%s. "
                "Reallocating buffer.",
                layer_index,
                buffer_name,
                existing_key.shape,
                shape,
                existing_key.dtype_name,
                dtype_name,
            )
            # Remove old buffer
            del self._buffers[existing_key]
            del self._lookup_cache[lookup_key]
            self._reallocation_count += 1

        # Allocate new buffer
        new_key = DecodeOutputBufferKey(
            layer_index=layer_index,
            buffer_name=buffer_name,
            shape=shape,
            dtype_name=dtype_name,
            device_name=device_name,
        )

        buffer = torch.empty(shape, dtype=dtype, device=device)
        self._buffers[new_key] = buffer
        self._lookup_cache[lookup_key] = new_key
        self._allocation_count += 1

        logger.debug(
            "Allocated decode output buffer: layer=%d, name=%r, "
            "dtype=%s, shape=%s, device=%s",
            layer_index,
            buffer_name,
            dtype_name,
            shape,
            device_name,
        )

        return buffer

    def clear(self) -> None:
        """Clear all buffers from the pool, releasing memory.

        Resets the pool to its initial empty state. Use this on request
        completion, cancellation, or when the decode microbatch layout changes
        and existing buffers are no longer valid.

        The allocation, reuse, and reallocation counters are preserved for
        lifetime statistics.
        """
        buffer_count = len(self._buffers)

        self._buffers.clear()
        self._lookup_cache.clear()

        logger.debug(
            "Decode output buffer pool cleared: freed %d buffers",
            buffer_count,
        )

    def clear_layer(self, layer_index: int) -> None:
        """Clear all buffers for a specific layer.

        Removes all buffers associated with the given layer index. Use this
        when a specific layer's state is being reset without affecting other
        layers.

        Args:
            layer_index: Global layer index whose buffers should be cleared.
        """
        keys_to_remove: list[DecodeOutputBufferKey] = [
            key for key in self._buffers if key.layer_index == layer_index
        ]

        for key in keys_to_remove:
            del self._buffers[key]
            lookup_key = (key.layer_index, key.buffer_name)
            self._lookup_cache.pop(lookup_key, None)

        if keys_to_remove:
            logger.debug(
                "Cleared %d decode output buffers for layer %d",
                len(keys_to_remove),
                layer_index,
            )

    def get_statistics(self) -> DecodeOutputBufferPoolStatistics:
        """Return a snapshot of the current pool state.

        Provides visibility into pool utilization for instrumentation,
        debugging, and performance monitoring.

        Returns:
            An immutable statistics snapshot with buffer counts and reuse metrics.
        """
        return DecodeOutputBufferPoolStatistics(
            total_buffer_count=len(self._buffers),
            allocation_count=self._allocation_count,
            reuse_count=self._reuse_count,
            reallocation_count=self._reallocation_count,
        )

    @property
    def buffer_count(self) -> int:
        """Number of buffers currently in the pool."""
        return len(self._buffers)

    @property
    def allocation_count(self) -> int:
        """Cumulative number of buffer allocations."""
        return self._allocation_count

    @property
    def reuse_count(self) -> int:
        """Cumulative number of buffer reuses."""
        return self._reuse_count

    @property
    def reallocation_count(self) -> int:
        """Cumulative number of buffer reallocations due to shape change."""
        return self._reallocation_count
