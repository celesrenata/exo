"""
Unit tests for DecodeOutputBufferPool.

Tests buffer allocation, reuse across decode steps, safe fallback on shape
change, layer-specific clearing, and statistics tracking.

**Validates: Requirements 5.3, 5.4**
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
_POOL_PATH = _THIS_DIR.parent / "decode_output_buffer_pool.py"


def _load_module() -> types.ModuleType:
    """Load decode_output_buffer_pool.py directly from file."""
    module_name = "decode_output_buffer_pool_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _POOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
DecodeOutputBufferPool = _mod.DecodeOutputBufferPool
DecodeOutputBufferKey = _mod.DecodeOutputBufferKey
DecodeOutputBufferPoolStatistics = _mod.DecodeOutputBufferPoolStatistics


# ===========================================================================
# Tests for DecodeOutputBufferKey
# ===========================================================================


class TestDecodeOutputBufferKey:
    """Test buffer key immutability and equality."""

    def test_key_is_frozen(self) -> None:
        """Buffer keys are immutable (frozen dataclass)."""
        key = DecodeOutputBufferKey(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        with pytest.raises(AttributeError):
            key.buffer_name = "modified"  # type: ignore[misc]

    def test_key_equality(self) -> None:
        """Keys with identical fields are equal."""
        key_a = DecodeOutputBufferKey(
            layer_index=3,
            buffer_name="gating_intermediate",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        key_b = DecodeOutputBufferKey(
            layer_index=3,
            buffer_name="gating_intermediate",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        assert key_a == key_b

    def test_key_inequality_different_layer(self) -> None:
        """Keys with different layer indices are not equal."""
        key_a = DecodeOutputBufferKey(
            layer_index=0,
            buffer_name="output_projection",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        key_b = DecodeOutputBufferKey(
            layer_index=1,
            buffer_name="output_projection",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        assert key_a != key_b

    def test_key_inequality_different_name(self) -> None:
        """Keys with different buffer names are not equal."""
        key_a = DecodeOutputBufferKey(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        key_b = DecodeOutputBufferKey(
            layer_index=0,
            buffer_name="gating_intermediate",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        assert key_a != key_b

    def test_key_hashable(self) -> None:
        """Keys can be used as dictionary keys."""
        key = DecodeOutputBufferKey(
            layer_index=5,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype_name="torch.bfloat16",
            device_name="cpu",
        )
        d: dict[DecodeOutputBufferKey, int] = {key: 42}
        assert d[key] == 42


# ===========================================================================
# Tests for DecodeOutputBufferPool — Allocation
# ===========================================================================


class TestDecodeOutputBufferPoolAllocation:
    """Test initial buffer allocation behavior."""

    def test_first_call_allocates_buffer(self) -> None:
        """First get_buffer call allocates a new tensor."""
        pool = DecodeOutputBufferPool()
        buffer = pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buffer.shape == (1, 1, 3584)
        assert buffer.dtype == torch.bfloat16
        assert buffer.device == torch.device("cpu")

    def test_allocation_count_increments(self) -> None:
        """Allocation count increments on first call."""
        pool = DecodeOutputBufferPool()
        assert pool.allocation_count == 0

        pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.allocation_count == 1

    def test_different_layers_get_different_buffers(self) -> None:
        """Different layer indices get independent buffers."""
        pool = DecodeOutputBufferPool()
        buf_0 = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_1 = pool.get_buffer(
            layer_index=1,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_0.data_ptr() != buf_1.data_ptr()
        assert pool.allocation_count == 2

    def test_different_names_same_layer_get_different_buffers(self) -> None:
        """Different buffer names on the same layer get independent buffers."""
        pool = DecodeOutputBufferPool()
        buf_rms = pool.get_buffer(
            layer_index=3,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_gate = pool.get_buffer(
            layer_index=3,
            buffer_name="gating_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_rms.data_ptr() != buf_gate.data_ptr()
        assert pool.allocation_count == 2

    def test_buffer_count_reflects_allocations(self) -> None:
        """buffer_count property reflects the number of allocated buffers."""
        pool = DecodeOutputBufferPool()
        assert pool.buffer_count == 0

        pool.get_buffer(
            layer_index=0,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.buffer_count == 1

        pool.get_buffer(
            layer_index=1,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.buffer_count == 2


# ===========================================================================
# Tests for DecodeOutputBufferPool — Reuse
# ===========================================================================


class TestDecodeOutputBufferPoolReuse:
    """Test buffer reuse across decode steps."""

    def test_same_shape_reuses_buffer(self) -> None:
        """Repeated calls with same shape return the same buffer (same data_ptr)."""
        pool = DecodeOutputBufferPool()
        buf_1 = pool.get_buffer(
            layer_index=5,
            buffer_name="output_projection",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_2 = pool.get_buffer(
            layer_index=5,
            buffer_name="output_projection",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_1.data_ptr() == buf_2.data_ptr()

    def test_reuse_count_increments(self) -> None:
        """Reuse count increments on subsequent calls with same shape."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.reuse_count == 0

        pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.reuse_count == 1

        pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.reuse_count == 2

    def test_allocation_count_does_not_increment_on_reuse(self) -> None:
        """Allocation count stays at 1 when buffer is reused."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.allocation_count == 1

    def test_multiple_decode_steps_reuse(self) -> None:
        """Simulates multiple decode steps — all reuse the same buffers."""
        pool = DecodeOutputBufferPool()
        shape = (1, 1, 3584)
        dtype = torch.bfloat16
        device = torch.device("cpu")

        # Simulate 10 decode steps across 4 buffer types for 2 layers
        buffer_names = [
            "layer_output",
            "rmsnorm_intermediate",
            "gating_intermediate",
            "output_projection",
        ]
        first_ptrs: dict[tuple[int, str], int] = {}

        for step in range(10):
            for layer_idx in range(2):
                for name in buffer_names:
                    buf = pool.get_buffer(
                        layer_index=layer_idx,
                        buffer_name=name,
                        shape=shape,
                        dtype=dtype,
                        device=device,
                    )
                    key = (layer_idx, name)
                    if step == 0:
                        first_ptrs[key] = buf.data_ptr()
                    else:
                        assert buf.data_ptr() == first_ptrs[key]

        # 2 layers × 4 buffers = 8 allocations
        assert pool.allocation_count == 8
        # 9 reuse steps × 2 layers × 4 buffers = 72 reuses
        assert pool.reuse_count == 72


# ===========================================================================
# Tests for DecodeOutputBufferPool — Shape Change Fallback
# ===========================================================================


class TestDecodeOutputBufferPoolShapeChange:
    """Test safe fallback on shape change (reallocation)."""

    def test_shape_change_reallocates(self) -> None:
        """When shape changes, a new buffer is allocated."""
        pool = DecodeOutputBufferPool()
        buf_1 = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        # Shape changes (batch size 1 → 4)
        buf_2 = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(4, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_1.data_ptr() != buf_2.data_ptr()
        assert buf_2.shape == (4, 1, 3584)

    def test_shape_change_increments_reallocation_count(self) -> None:
        """Reallocation count increments on shape change."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.reallocation_count == 0

        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(2, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.reallocation_count == 1

    def test_shape_change_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Shape change logs a warning message."""
        import logging

        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=3,
            buffer_name="gating_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        with caplog.at_level(logging.WARNING):
            pool.get_buffer(
                layer_index=3,
                buffer_name="gating_intermediate",
                shape=(2, 1, 3584),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

        assert "shape changed" in caplog.text.lower()
        assert "layer=3" in caplog.text

    def test_after_shape_change_new_shape_is_reused(self) -> None:
        """After reallocation, the new shape is reused on subsequent calls."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_new = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(4, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_reused = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(4, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_new.data_ptr() == buf_reused.data_ptr()
        assert pool.reuse_count == 1

    def test_dtype_change_reallocates(self) -> None:
        """When dtype changes, a new buffer is allocated."""
        pool = DecodeOutputBufferPool()
        buf_bf16 = pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_fp32 = pool.get_buffer(
            layer_index=0,
            buffer_name="rmsnorm_intermediate",
            shape=(1, 1, 3584),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert buf_bf16.data_ptr() != buf_fp32.data_ptr()
        assert buf_fp32.dtype == torch.float32
        assert pool.reallocation_count == 1

    def test_buffer_count_stays_same_after_reallocation(self) -> None:
        """Buffer count does not increase on reallocation (old is removed)."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.buffer_count == 1

        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(4, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.buffer_count == 1


# ===========================================================================
# Tests for DecodeOutputBufferPool — Clear
# ===========================================================================


class TestDecodeOutputBufferPoolClear:
    """Test pool clearing behavior."""

    def test_clear_removes_all_buffers(self) -> None:
        """clear() removes all buffers from the pool."""
        pool = DecodeOutputBufferPool()
        for i in range(5):
            pool.get_buffer(
                layer_index=i,
                buffer_name="layer_output",
                shape=(1, 1, 3584),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
        assert pool.buffer_count == 5

        pool.clear()
        assert pool.buffer_count == 0

    def test_clear_preserves_statistics(self) -> None:
        """clear() preserves allocation and reuse counters."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=0,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        pool.clear()

        assert pool.allocation_count == 1
        assert pool.reuse_count == 1

    def test_after_clear_next_call_allocates_fresh(self) -> None:
        """After clear(), the next get_buffer call allocates a new buffer."""
        pool = DecodeOutputBufferPool()
        buf_1 = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        ptr_1 = buf_1.data_ptr()

        pool.clear()

        buf_2 = pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        # New allocation — different data pointer
        assert buf_2.data_ptr() != ptr_1
        assert pool.allocation_count == 2

    def test_clear_layer_removes_only_target_layer(self) -> None:
        """clear_layer() removes only buffers for the specified layer."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=0,
            buffer_name="b",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=1,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert pool.buffer_count == 3

        pool.clear_layer(0)
        assert pool.buffer_count == 1

    def test_clear_layer_does_not_affect_other_layers(self) -> None:
        """clear_layer() leaves other layers' buffers intact."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        buf_layer_1 = pool.get_buffer(
            layer_index=1,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        ptr_layer_1 = buf_layer_1.data_ptr()

        pool.clear_layer(0)

        # Layer 1 buffer still reusable
        buf_layer_1_again = pool.get_buffer(
            layer_index=1,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        assert buf_layer_1_again.data_ptr() == ptr_layer_1

    def test_clear_layer_nonexistent_is_noop(self) -> None:
        """clear_layer() on a layer with no buffers does nothing."""
        pool = DecodeOutputBufferPool()
        pool.get_buffer(
            layer_index=0,
            buffer_name="a",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.clear_layer(99)  # No buffers for layer 99
        assert pool.buffer_count == 1


# ===========================================================================
# Tests for DecodeOutputBufferPool — Statistics
# ===========================================================================


class TestDecodeOutputBufferPoolStatistics:
    """Test statistics reporting."""

    def test_initial_statistics_are_zero(self) -> None:
        """Fresh pool has all-zero statistics."""
        pool = DecodeOutputBufferPool()
        stats = pool.get_statistics()
        assert stats.total_buffer_count == 0
        assert stats.allocation_count == 0
        assert stats.reuse_count == 0
        assert stats.reallocation_count == 0

    def test_statistics_reflect_operations(self) -> None:
        """Statistics accurately reflect pool operations."""
        pool = DecodeOutputBufferPool()

        # Allocate 3 buffers
        for i in range(3):
            pool.get_buffer(
                layer_index=i,
                buffer_name="layer_output",
                shape=(1, 1, 3584),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )

        # Reuse 2 of them
        pool.get_buffer(
            layer_index=0,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        pool.get_buffer(
            layer_index=1,
            buffer_name="layer_output",
            shape=(1, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        # Reallocate 1 (shape change)
        pool.get_buffer(
            layer_index=2,
            buffer_name="layer_output",
            shape=(4, 1, 3584),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        stats = pool.get_statistics()
        assert stats.total_buffer_count == 3
        assert stats.allocation_count == 4  # 3 initial + 1 reallocation
        assert stats.reuse_count == 2
        assert stats.reallocation_count == 1

    def test_statistics_is_frozen(self) -> None:
        """Statistics snapshot is immutable."""
        pool = DecodeOutputBufferPool()
        stats = pool.get_statistics()
        with pytest.raises(AttributeError):
            stats.total_buffer_count = 99  # type: ignore[misc]
