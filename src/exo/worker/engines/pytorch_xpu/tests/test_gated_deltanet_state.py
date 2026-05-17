"""
Unit tests for GatedDeltaNetPersistentState container.

Tests state creation, reset, recycle, claim lifecycle, shape validation,
dtype persistence, and output buffer preallocation.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.7, 5.8**
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
_STATE_MODULE_PATH = _THIS_DIR.parent / "gated_deltanet_state.py"


def _load_state_module() -> types.ModuleType:
    """Load gated_deltanet_state.py directly from file, avoiding __init__.py."""
    module_name = "gated_deltanet_state_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _STATE_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_state_module()
GatedDeltaNetPersistentState = _mod.GatedDeltaNetPersistentState
GatedDeltaNetStateShape = _mod.GatedDeltaNetStateShape


# ===========================================================================
# Test fixtures
# ===========================================================================

# Qwen3.5-4B dimensions for testing
_BATCH_SIZE = 1
_NUM_HEADS = 32
_KEY_DIM = 128
_VALUE_DIM = 128
_CONV_DIM = 256
_CONV_KERNEL_SIZE = 4


@pytest.fixture
def persistent_state() -> "GatedDeltaNetPersistentState":
    """Create a persistent state container with Qwen3.5-4B dimensions."""
    return GatedDeltaNetPersistentState.create(
        request_identifier="test-request-001",
        layer_index=5,
        batch_size=_BATCH_SIZE,
        num_heads=_NUM_HEADS,
        key_dim=_KEY_DIM,
        value_dim=_VALUE_DIM,
        conv_dim=_CONV_DIM,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        device=torch.device("cpu"),
    )


# ===========================================================================
# Tests for GatedDeltaNetStateShape
# ===========================================================================


class TestGatedDeltaNetStateShape:
    """Test shape metadata immutability and computed properties."""

    def test_shape_is_frozen(self) -> None:
        """Shape metadata is immutable."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        with pytest.raises(AttributeError):
            shape.batch_size = 2  # type: ignore[misc]

    def test_recurrent_state_shape(self) -> None:
        """Recurrent state shape is (B, H, d_k, d_v)."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        assert shape.recurrent_state_shape == (1, 32, 128, 128)

    def test_conv_state_shape(self) -> None:
        """Conv state shape is (B, conv_dim, kernel_size)."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        assert shape.conv_state_shape == (1, 256, 4)

    def test_output_shape(self) -> None:
        """Output shape is (B, H, d_v)."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        assert shape.output_shape == (1, 32, 128)

    def test_element_count(self) -> None:
        """Element count matches product of dimensions."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        assert shape.recurrent_state_element_count == 1 * 32 * 128 * 128

    def test_bytes_fp32(self) -> None:
        """Byte count is element_count * 4 for fp32."""
        shape = GatedDeltaNetStateShape(
            batch_size=1,
            num_heads=32,
            key_dim=128,
            value_dim=128,
            conv_dim=256,
            conv_kernel_size=4,
        )
        assert shape.recurrent_state_bytes_fp32 == 1 * 32 * 128 * 128 * 4


# ===========================================================================
# Tests for GatedDeltaNetPersistentState creation
# ===========================================================================


class TestPersistentStateCreation:
    """Test state container creation and initial tensor properties."""

    def test_create_stores_request_identifier(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Created state stores the request identifier."""
        assert persistent_state.request_identifier == "test-request-001"

    def test_create_stores_layer_index(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Created state stores the layer index."""
        assert persistent_state.layer_index == 5

    def test_recurrent_state_dtype_is_fp32(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state is stored in fp32 for numerical stability."""
        assert persistent_state.recurrent_state.dtype == torch.float32

    def test_recurrent_state_shape(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state has shape (B, H, d_k, d_v)."""
        assert persistent_state.recurrent_state.shape == (
            _BATCH_SIZE,
            _NUM_HEADS,
            _KEY_DIM,
            _VALUE_DIM,
        )

    def test_conv_state_dtype_is_bf16(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Conv state is stored in bf16 (model compute dtype)."""
        assert persistent_state.conv_state.dtype == torch.bfloat16

    def test_conv_state_shape(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Conv state has shape (B, conv_dim, kernel_size)."""
        assert persistent_state.conv_state.shape == (
            _BATCH_SIZE,
            _CONV_DIM,
            _CONV_KERNEL_SIZE,
        )

    def test_output_buffer_fp32_dtype(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output buffer fp32 has correct dtype."""
        assert persistent_state.output_buffer_fp32.dtype == torch.float32

    def test_output_buffer_bf16_dtype(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output buffer bf16 has correct dtype."""
        assert persistent_state.output_buffer_bf16.dtype == torch.bfloat16

    def test_output_buffers_shape(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output buffers have shape (B, H, d_v)."""
        expected = (_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM)
        assert persistent_state.output_buffer_fp32.shape == expected
        assert persistent_state.output_buffer_bf16.shape == expected

    def test_device_stored(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Device reference is stored."""
        assert persistent_state.device == torch.device("cpu")

    def test_initial_state_is_zero(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """All tensors are initialized to zero."""
        assert torch.all(persistent_state.recurrent_state == 0)
        assert torch.all(persistent_state.conv_state == 0)
        assert torch.all(persistent_state.output_buffer_fp32 == 0)
        assert torch.all(persistent_state.output_buffer_bf16 == 0)

    def test_not_available_for_reuse_initially(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Newly created state is not available for reuse."""
        assert not persistent_state.is_available_for_reuse

    def test_decode_step_count_starts_at_zero(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Decode step counter starts at zero."""
        assert persistent_state.decode_step_count == 0


# ===========================================================================
# Tests for reset method
# ===========================================================================


class TestPersistentStateReset:
    """Test reset zeros tensors without reallocation."""

    def test_reset_zeros_recurrent_state(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset zeros the recurrent state tensor."""
        # Modify state
        persistent_state.recurrent_state.fill_(42.0)
        persistent_state.reset()
        assert torch.all(persistent_state.recurrent_state == 0)

    def test_reset_zeros_conv_state(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset zeros the conv state tensor."""
        persistent_state.conv_state.fill_(1.0)
        persistent_state.reset()
        assert torch.all(persistent_state.conv_state == 0)

    def test_reset_zeros_output_buffers(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset zeros both output buffers."""
        persistent_state.output_buffer_fp32.fill_(1.0)
        persistent_state.output_buffer_bf16.fill_(1.0)
        persistent_state.reset()
        assert torch.all(persistent_state.output_buffer_fp32 == 0)
        assert torch.all(persistent_state.output_buffer_bf16 == 0)

    def test_reset_preserves_tensor_identity(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset does not reallocate — same tensor object (same data_ptr)."""
        recurrent_ptr = persistent_state.recurrent_state.data_ptr()
        conv_ptr = persistent_state.conv_state.data_ptr()
        output_fp32_ptr = persistent_state.output_buffer_fp32.data_ptr()
        output_bf16_ptr = persistent_state.output_buffer_bf16.data_ptr()

        persistent_state.recurrent_state.fill_(99.0)
        persistent_state.reset()

        assert persistent_state.recurrent_state.data_ptr() == recurrent_ptr
        assert persistent_state.conv_state.data_ptr() == conv_ptr
        assert persistent_state.output_buffer_fp32.data_ptr() == output_fp32_ptr
        assert persistent_state.output_buffer_bf16.data_ptr() == output_bf16_ptr

    def test_reset_preserves_request_identifier(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset keeps the request identifier unchanged."""
        persistent_state.reset()
        assert persistent_state.request_identifier == "test-request-001"

    def test_reset_resets_decode_step_count(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset resets the decode step counter."""
        persistent_state.increment_decode_step()
        persistent_state.increment_decode_step()
        assert persistent_state.decode_step_count == 2
        persistent_state.reset()
        assert persistent_state.decode_step_count == 0

    def test_reset_clears_available_flag(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Reset marks the state as not available for reuse."""
        persistent_state.recycle()
        assert persistent_state.is_available_for_reuse
        persistent_state.claim("new-request")
        persistent_state.reset()
        assert not persistent_state.is_available_for_reuse


# ===========================================================================
# Tests for recycle method
# ===========================================================================


class TestPersistentStateRecycle:
    """Test recycle marks state as available and clears ownership."""

    def test_recycle_marks_available(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recycle marks the container as available for reuse."""
        persistent_state.recycle()
        assert persistent_state.is_available_for_reuse

    def test_recycle_clears_request_identifier(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recycle clears the request identifier."""
        persistent_state.recycle()
        assert persistent_state.request_identifier == ""

    def test_recycle_zeros_tensors(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recycle zeros all state tensors."""
        persistent_state.recurrent_state.fill_(42.0)
        persistent_state.conv_state.fill_(1.0)
        persistent_state.recycle()
        assert torch.all(persistent_state.recurrent_state == 0)
        assert torch.all(persistent_state.conv_state == 0)

    def test_recycle_preserves_tensor_identity(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recycle does not reallocate — same underlying memory."""
        recurrent_ptr = persistent_state.recurrent_state.data_ptr()
        persistent_state.recurrent_state.fill_(99.0)
        persistent_state.recycle()
        assert persistent_state.recurrent_state.data_ptr() == recurrent_ptr

    def test_recycle_resets_decode_step_count(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recycle resets the decode step counter."""
        persistent_state.increment_decode_step()
        persistent_state.recycle()
        assert persistent_state.decode_step_count == 0


# ===========================================================================
# Tests for claim method
# ===========================================================================


class TestPersistentStateClaim:
    """Test claiming a recycled container for a new request."""

    def test_claim_sets_request_identifier(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Claim assigns the new request identifier."""
        persistent_state.recycle()
        persistent_state.claim("new-request-002")
        assert persistent_state.request_identifier == "new-request-002"

    def test_claim_marks_not_available(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Claim marks the container as no longer available."""
        persistent_state.recycle()
        persistent_state.claim("new-request-002")
        assert not persistent_state.is_available_for_reuse

    def test_claim_raises_if_not_recycled(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Claim raises RuntimeError if the container is not recycled."""
        with pytest.raises(RuntimeError, match="not recycled"):
            persistent_state.claim("should-fail")


# ===========================================================================
# Tests for shape validation
# ===========================================================================


class TestPersistentStateShapeValidation:
    """Test shape validation against expected dimensions."""

    def test_validate_matching_shape(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Validation passes for matching dimensions."""
        assert persistent_state.validate_shape(
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
        )

    def test_validate_mismatched_batch_size(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Validation fails for mismatched batch size."""
        assert not persistent_state.validate_shape(
            batch_size=2,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
        )

    def test_validate_mismatched_num_heads(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Validation fails for mismatched num_heads."""
        assert not persistent_state.validate_shape(
            batch_size=_BATCH_SIZE,
            num_heads=64,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
        )

    def test_validate_mismatched_key_dim(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Validation fails for mismatched key_dim."""
        assert not persistent_state.validate_shape(
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=64,
            value_dim=_VALUE_DIM,
        )


# ===========================================================================
# Tests for decode step counting
# ===========================================================================


class TestDecodeStepCounting:
    """Test decode step counter increments correctly."""

    def test_increment_increases_count(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Each increment increases the count by one."""
        persistent_state.increment_decode_step()
        assert persistent_state.decode_step_count == 1
        persistent_state.increment_decode_step()
        assert persistent_state.decode_step_count == 2
        persistent_state.increment_decode_step()
        assert persistent_state.decode_step_count == 3


# ===========================================================================
# Tests for dtype persistence (Requirement 5.5)
# ===========================================================================


class TestDtypePersistence:
    """Test that fp32 dtype is maintained across operations."""

    def test_recurrent_state_stays_fp32_after_modification(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state remains fp32 after in-place modification."""
        persistent_state.recurrent_state.add_(torch.randn_like(persistent_state.recurrent_state))
        assert persistent_state.recurrent_state.dtype == torch.float32

    def test_recurrent_state_stays_fp32_after_reset(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state remains fp32 after reset."""
        persistent_state.recurrent_state.fill_(1.0)
        persistent_state.reset()
        assert persistent_state.recurrent_state.dtype == torch.float32

    def test_recurrent_state_stays_fp32_after_recycle(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state remains fp32 after recycle."""
        persistent_state.recurrent_state.fill_(1.0)
        persistent_state.recycle()
        assert persistent_state.recurrent_state.dtype == torch.float32


# ===========================================================================
# Tests for repr
# ===========================================================================


class TestRepr:
    """Test string representation for debugging."""

    def test_repr_contains_request(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Repr includes request identifier."""
        assert "test-request-001" in repr(persistent_state)

    def test_repr_contains_layer(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Repr includes layer index."""
        assert "layer=5" in repr(persistent_state)

    def test_repr_shows_active_status(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Repr shows active status for non-recycled state."""
        assert "status=active" in repr(persistent_state)

    def test_repr_shows_available_status(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Repr shows available status after recycle."""
        persistent_state.recycle()
        assert "status=available" in repr(persistent_state)
