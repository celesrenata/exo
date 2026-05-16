"""
Unit tests for CommunicationBufferPool.

Tests buffer acquisition, reuse, release, clearing, and statistics tracking.

**Validates: Requirements 2.3, 2.4, 2.6**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BUFFER_POOL_PATH = _THIS_DIR.parent / "buffer_pool.py"


def _load_buffer_pool() -> types.ModuleType:
    """Load buffer_pool.py directly from file, avoiding __init__.py."""
    module_name = "buffer_pool_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _BUFFER_POOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_buffer_pool()
CommunicationBufferPool = _mod.CommunicationBufferPool
CommunicationBufferKey = _mod.CommunicationBufferKey
BufferPoolStatistics = _mod.BufferPoolStatistics


# ===========================================================================
# Tests for CommunicationBufferKey
# ===========================================================================


class TestCommunicationBufferKey:
    """Test buffer key immutability and hashing."""

    def test_key_is_frozen(self) -> None:
        """Buffer keys are immutable."""
        from pydantic import ValidationError

        key = CommunicationBufferKey(
            buffer_name="test",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        with pytest.raises(ValidationError):
            key.buffer_name = "modified"  # type: ignore[misc]

    def test_key_equality(self) -> None:
        """Keys with identical fields are equal."""
        key_a = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        key_b = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        assert key_a == key_b

    def test_key_inequality_on_shape(self) -> None:
        """Keys with different shapes are not equal."""
        key_a = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        key_b = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(2, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        assert key_a != key_b

    def test_key_inequality_on_rank(self) -> None:
        """Keys with different ranks are not equal."""
        key_a = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        key_b = CommunicationBufferKey(
            buffer_name="activation",
            dtype_name="torch.bfloat16",
            shape=(1, 1, 3584),
            source_rank=1,
            destination_rank=2,
        )
        assert key_a != key_b

    def test_key_is_hashable(self) -> None:
        """Buffer keys can be used as dictionary keys."""
        key = CommunicationBufferKey(
            buffer_name="test",
            dtype_name="torch.float32",
            shape=(4, 1, 2048),
            source_rank=2,
            destination_rank=3,
        )
        mapping: dict[CommunicationBufferKey, int] = {key: 42}
        assert mapping[key] == 42


# ===========================================================================
# Tests for BufferPoolStatistics
# ===========================================================================


class TestBufferPoolStatistics:
    """Test statistics model immutability."""

    def test_statistics_is_frozen(self) -> None:
        """Statistics snapshots are immutable."""
        from pydantic import ValidationError

        stats = BufferPoolStatistics(
            total_buffer_count=5,
            active_buffer_count=2,
            available_buffer_count=3,
            reuse_count=10,
            allocation_count=5,
        )
        with pytest.raises(ValidationError):
            stats.total_buffer_count = 99  # type: ignore[misc]


# ===========================================================================
# Tests for CommunicationBufferPool — acquisition and allocation
# ===========================================================================


class TestBufferPoolAcquisition:
    """Test buffer acquisition allocates correctly.

    **Validates: Requirements 2.3, 2.4**
    """

    def test_acquire_allocates_cpu_tensor(self) -> None:
        """First acquisition allocates a new CPU tensor."""
        pool = CommunicationBufferPool()
        buffer = pool.acquire(
            buffer_name="send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        assert isinstance(buffer, torch.Tensor)
        assert buffer.device.type == "cpu"
        assert buffer.dtype == torch.bfloat16
        assert tuple(buffer.shape) == (1, 1, 3584)

    def test_acquire_with_different_dtypes(self) -> None:
        """Buffers with different dtypes are distinct."""
        pool = CommunicationBufferPool()
        bf16_buffer = pool.acquire(
            buffer_name="activation",
            dtype=torch.bfloat16,
            shape=(1, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="activation",
            dtype=torch.bfloat16,
            shape=(1, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )

        fp32_buffer = pool.acquire(
            buffer_name="activation",
            dtype=torch.float32,
            shape=(1, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )

        assert bf16_buffer.dtype == torch.bfloat16
        assert fp32_buffer.dtype == torch.float32
        assert bf16_buffer is not fp32_buffer

    def test_acquire_with_different_shapes(self) -> None:
        """Buffers with different shapes are distinct."""
        pool = CommunicationBufferPool()
        small_buffer = pool.acquire(
            buffer_name="activation",
            dtype=torch.bfloat16,
            shape=(1, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="activation",
            dtype=torch.bfloat16,
            shape=(1, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )

        large_buffer = pool.acquire(
            buffer_name="activation",
            dtype=torch.bfloat16,
            shape=(4, 1, 2048),
            source_rank=0,
            destination_rank=1,
        )

        assert tuple(small_buffer.shape) == (1, 1, 2048)
        assert tuple(large_buffer.shape) == (4, 1, 2048)
        assert small_buffer is not large_buffer

    def test_acquire_raises_on_double_acquire(self) -> None:
        """Acquiring the same buffer twice without release raises RuntimeError."""
        pool = CommunicationBufferPool()
        pool.acquire(
            buffer_name="send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        with pytest.raises(RuntimeError, match="already acquired"):
            pool.acquire(
                buffer_name="send",
                dtype=torch.bfloat16,
                shape=(1, 1, 3584),
                source_rank=0,
                destination_rank=1,
            )


# ===========================================================================
# Tests for CommunicationBufferPool — reuse
# ===========================================================================


class TestBufferPoolReuse:
    """Test buffer reuse across decode steps.

    **Validates: Requirements 2.3, 2.4, 2.6**
    """

    def test_reuse_returns_same_tensor(self) -> None:
        """After release, acquiring the same key returns the same tensor object."""
        pool = CommunicationBufferPool()

        # First acquire
        buffer_first = pool.acquire(
            buffer_name="decode_send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        buffer_first_data_ptr = buffer_first.data_ptr()

        # Release
        pool.release(
            buffer_name="decode_send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        # Second acquire — same key
        buffer_second = pool.acquire(
            buffer_name="decode_send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        assert buffer_second is buffer_first
        assert buffer_second.data_ptr() == buffer_first_data_ptr

    def test_reuse_increments_reuse_count(self) -> None:
        """Each reuse increments the reuse counter."""
        pool = CommunicationBufferPool()

        for _ in range(5):
            pool.acquire(
                buffer_name="recv",
                dtype=torch.float32,
                shape=(1, 1, 2048),
                source_rank=1,
                destination_rank=2,
            )
            pool.release(
                buffer_name="recv",
                dtype=torch.float32,
                shape=(1, 1, 2048),
                source_rank=1,
                destination_rank=2,
            )

        stats = pool.get_statistics()
        assert stats.allocation_count == 1  # Only one allocation
        assert stats.reuse_count == 4  # First acquire is allocation, next 4 are reuses

    def test_multiple_buffers_independent_reuse(self) -> None:
        """Multiple distinct buffers can be acquired and reused independently."""
        pool = CommunicationBufferPool()

        # Acquire two different buffers
        send_buf = pool.acquire(
            buffer_name="send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        recv_buf = pool.acquire(
            buffer_name="recv",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        assert send_buf is not recv_buf

        # Release both
        pool.release(
            buffer_name="send",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="recv",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        stats = pool.get_statistics()
        assert stats.total_buffer_count == 2
        assert stats.active_buffer_count == 0
        assert stats.available_buffer_count == 2


# ===========================================================================
# Tests for CommunicationBufferPool — release
# ===========================================================================


class TestBufferPoolRelease:
    """Test buffer release behavior."""

    def test_release_makes_buffer_available(self) -> None:
        """Released buffer transitions from active to available."""
        pool = CommunicationBufferPool()
        pool.acquire(
            buffer_name="test",
            dtype=torch.float32,
            shape=(2, 1, 1024),
            source_rank=0,
            destination_rank=1,
        )

        stats_before = pool.get_statistics()
        assert stats_before.active_buffer_count == 1
        assert stats_before.available_buffer_count == 0

        pool.release(
            buffer_name="test",
            dtype=torch.float32,
            shape=(2, 1, 1024),
            source_rank=0,
            destination_rank=1,
        )

        stats_after = pool.get_statistics()
        assert stats_after.active_buffer_count == 0
        assert stats_after.available_buffer_count == 1

    def test_release_raises_on_unacquired_buffer(self) -> None:
        """Releasing a buffer that was not acquired raises RuntimeError."""
        pool = CommunicationBufferPool()
        with pytest.raises(RuntimeError, match="not acquired"):
            pool.release(
                buffer_name="nonexistent",
                dtype=torch.float32,
                shape=(1, 1, 512),
                source_rank=0,
                destination_rank=1,
            )

    def test_double_release_raises(self) -> None:
        """Releasing the same buffer twice raises RuntimeError."""
        pool = CommunicationBufferPool()
        pool.acquire(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )
        with pytest.raises(RuntimeError, match="not acquired"):
            pool.release(
                buffer_name="test",
                dtype=torch.float32,
                shape=(1, 1, 512),
                source_rank=0,
                destination_rank=1,
            )


# ===========================================================================
# Tests for CommunicationBufferPool — clear
# ===========================================================================


class TestBufferPoolClear:
    """Test pool clearing behavior."""

    def test_clear_removes_all_buffers(self) -> None:
        """Clear deallocates all buffers."""
        pool = CommunicationBufferPool()
        pool.acquire(
            buffer_name="a",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="a",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        pool.acquire(
            buffer_name="b",
            dtype=torch.float32,
            shape=(1, 1, 2048),
            source_rank=1,
            destination_rank=2,
        )
        pool.release(
            buffer_name="b",
            dtype=torch.float32,
            shape=(1, 1, 2048),
            source_rank=1,
            destination_rank=2,
        )

        pool.clear()

        stats = pool.get_statistics()
        assert stats.total_buffer_count == 0
        assert stats.active_buffer_count == 0
        assert stats.available_buffer_count == 0

    def test_clear_preserves_lifetime_counters(self) -> None:
        """Clear preserves reuse and allocation counters for lifetime statistics."""
        pool = CommunicationBufferPool()

        # Allocate and reuse
        pool.acquire(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )
        pool.acquire(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="test",
            dtype=torch.float32,
            shape=(1, 1, 512),
            source_rank=0,
            destination_rank=1,
        )

        pool.clear()

        stats = pool.get_statistics()
        assert stats.allocation_count == 1
        assert stats.reuse_count == 1

    def test_clear_allows_fresh_allocation(self) -> None:
        """After clear, acquiring the same key allocates a new tensor."""
        pool = CommunicationBufferPool()

        buffer_before = pool.acquire(
            buffer_name="test",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        pool.release(
            buffer_name="test",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        pool.clear()

        buffer_after = pool.acquire(
            buffer_name="test",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )

        # New allocation — different tensor object
        assert buffer_after is not buffer_before

    def test_clear_with_active_buffers(self) -> None:
        """Clear forcibly releases active buffers."""
        pool = CommunicationBufferPool()
        pool.acquire(
            buffer_name="active",
            dtype=torch.float32,
            shape=(1, 1, 1024),
            source_rank=0,
            destination_rank=1,
        )

        # Clear while buffer is still active
        pool.clear()

        stats = pool.get_statistics()
        assert stats.total_buffer_count == 0
        assert stats.active_buffer_count == 0


# ===========================================================================
# Tests for CommunicationBufferPool — statistics
# ===========================================================================


class TestBufferPoolStatisticsReporting:
    """Test pool statistics reporting.

    **Validates: Requirements 2.6**
    """

    def test_initial_statistics_are_zero(self) -> None:
        """Fresh pool has all-zero statistics."""
        pool = CommunicationBufferPool()
        stats = pool.get_statistics()
        assert stats.total_buffer_count == 0
        assert stats.active_buffer_count == 0
        assert stats.available_buffer_count == 0
        assert stats.reuse_count == 0
        assert stats.allocation_count == 0

    def test_statistics_after_acquisitions(self) -> None:
        """Statistics reflect acquisition state."""
        pool = CommunicationBufferPool()

        pool.acquire(
            buffer_name="send_0_1",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=0,
            destination_rank=1,
        )
        pool.acquire(
            buffer_name="send_1_2",
            dtype=torch.bfloat16,
            shape=(1, 1, 3584),
            source_rank=1,
            destination_rank=2,
        )

        stats = pool.get_statistics()
        assert stats.total_buffer_count == 2
        assert stats.active_buffer_count == 2
        assert stats.available_buffer_count == 0
        assert stats.allocation_count == 2
        assert stats.reuse_count == 0

    def test_statistics_track_reuse_across_decode_steps(self) -> None:
        """Statistics correctly track reuse simulating multiple decode steps."""
        pool = CommunicationBufferPool()

        # Simulate 10 decode steps with the same buffer
        for _step in range(10):
            pool.acquire(
                buffer_name="decode_activation",
                dtype=torch.bfloat16,
                shape=(1, 1, 3584),
                source_rank=0,
                destination_rank=1,
            )
            pool.release(
                buffer_name="decode_activation",
                dtype=torch.bfloat16,
                shape=(1, 1, 3584),
                source_rank=0,
                destination_rank=1,
            )

        stats = pool.get_statistics()
        assert stats.allocation_count == 1  # Only allocated once
        assert stats.reuse_count == 9  # Reused 9 times
        assert stats.total_buffer_count == 1
        assert stats.active_buffer_count == 0
        assert stats.available_buffer_count == 1
