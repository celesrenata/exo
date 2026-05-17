"""
Unit tests for initialize_gated_deltanet_state function.

Tests:
- State is initialized in fp32
- State is on the correct device
- Repeated calls for the same request/layer return the same state (idempotent)
- Recycled state is reused rather than reallocated (same data_ptr)
- Different requests get independent states

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
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_state_mod = _load_module(
    "gated_deltanet_state_init_test",
    _ENGINE_DIR / "gated_deltanet_state.py",
)
_cache_mod = _load_module(
    "gated_deltanet_cache_init_test",
    _ENGINE_DIR / "gated_deltanet_cache.py",
)

GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
initialize_gated_deltanet_state = _state_mod.initialize_gated_deltanet_state
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache


# ===========================================================================
# Fixtures
# ===========================================================================

# Qwen3.5-4B dimensions for testing
_BATCH_SIZE = 1
_NUM_HEADS = 32
_KEY_DIM = 128
_VALUE_DIM = 128
_CONV_DIM = 256
_CONV_KERNEL_SIZE = 4


class FakeDynamicCache:
    """Minimal DynamicCache mock for testing."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._seq_length: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_length

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.key_cache[index], self.value_cache[index])

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(len(self.key_cache)):
            yield (self.key_cache[i], self.value_cache[i])

    def __repr__(self) -> str:
        return f"FakeDynamicCache(layers={len(self.key_cache)}, seq_len={self._seq_length})"


@pytest.fixture
def cache() -> "GatedDeltaNetCache":
    """Create a GatedDeltaNetCache wrapping a fake DynamicCache."""
    return GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())


def _init_state(
    cache: "GatedDeltaNetCache",
    request_identifier: str = "request-001",
    layer_index: int = 5,
    device: torch.device | None = None,
) -> "GatedDeltaNetPersistentState":
    """Helper to call initialize_gated_deltanet_state with standard params."""
    return initialize_gated_deltanet_state(
        cache=cache,
        request_identifier=request_identifier,
        layer_index=layer_index,
        batch_size=_BATCH_SIZE,
        num_heads=_NUM_HEADS,
        key_dim=_KEY_DIM,
        value_dim=_VALUE_DIM,
        conv_dim=_CONV_DIM,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        device=device or torch.device("cpu"),
    )


# ===========================================================================
# Tests: State initialization in fp32
# ===========================================================================


class TestStateInitializedInFp32:
    """Verify that initialized state uses fp32 for recurrent state."""

    def test_recurrent_state_is_fp32(self, cache: "GatedDeltaNetCache") -> None:
        """Recurrent state tensor is allocated in fp32."""
        state = _init_state(cache)
        assert state.recurrent_state.dtype == torch.float32

    def test_output_buffer_fp32_is_fp32(self, cache: "GatedDeltaNetCache") -> None:
        """Output buffer fp32 is allocated in fp32."""
        state = _init_state(cache)
        assert state.output_buffer_fp32.dtype == torch.float32

    def test_conv_state_is_bf16(self, cache: "GatedDeltaNetCache") -> None:
        """Conv state is allocated in bf16 (model compute dtype)."""
        state = _init_state(cache)
        assert state.conv_state.dtype == torch.bfloat16

    def test_recurrent_state_shape_correct(self, cache: "GatedDeltaNetCache") -> None:
        """Recurrent state has expected shape (B, H, d_k, d_v)."""
        state = _init_state(cache)
        assert state.recurrent_state.shape == (
            _BATCH_SIZE,
            _NUM_HEADS,
            _KEY_DIM,
            _VALUE_DIM,
        )

    def test_initial_state_is_zero(self, cache: "GatedDeltaNetCache") -> None:
        """Newly initialized state tensors are zeroed."""
        state = _init_state(cache)
        assert torch.all(state.recurrent_state == 0)
        assert torch.all(state.conv_state == 0)
        assert torch.all(state.output_buffer_fp32 == 0)
        assert torch.all(state.output_buffer_bf16 == 0)


# ===========================================================================
# Tests: State is on the correct device
# ===========================================================================


class TestStateOnCorrectDevice:
    """Verify that state tensors are allocated on the specified device."""

    def test_state_device_is_cpu(self, cache: "GatedDeltaNetCache") -> None:
        """State tensors are on CPU when CPU device is specified."""
        state = _init_state(cache, device=torch.device("cpu"))
        assert state.device == torch.device("cpu")
        assert state.recurrent_state.device == torch.device("cpu")
        assert state.conv_state.device == torch.device("cpu")
        assert state.output_buffer_fp32.device == torch.device("cpu")
        assert state.output_buffer_bf16.device == torch.device("cpu")

    def test_state_device_stored_correctly(self, cache: "GatedDeltaNetCache") -> None:
        """The device attribute matches the device passed at initialization."""
        device = torch.device("cpu")
        state = _init_state(cache, device=device)
        assert state.device == device


# ===========================================================================
# Tests: Idempotent initialization (same request/layer returns same state)
# ===========================================================================


class TestIdempotentInitialization:
    """Verify that repeated calls for the same request/layer are idempotent."""

    def test_same_request_same_layer_returns_same_object(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Calling initialize twice for the same request/layer returns the same state."""
        state_1 = _init_state(cache, request_identifier="req-A", layer_index=3)
        state_2 = _init_state(cache, request_identifier="req-A", layer_index=3)
        assert state_1 is state_2

    def test_same_request_same_layer_same_data_ptr(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Idempotent calls preserve the same underlying memory (data_ptr)."""
        state_1 = _init_state(cache, request_identifier="req-A", layer_index=3)
        ptr_1 = state_1.recurrent_state.data_ptr()
        state_2 = _init_state(cache, request_identifier="req-A", layer_index=3)
        ptr_2 = state_2.recurrent_state.data_ptr()
        assert ptr_1 == ptr_2

    def test_idempotent_after_decode_steps(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Idempotent return works even after decode steps have been processed."""
        state = _init_state(cache, request_identifier="req-A", layer_index=7)
        state.increment_decode_step()
        state.increment_decode_step()
        state.recurrent_state.fill_(3.14)

        # Re-initialize — should return the same state with modifications intact
        state_again = _init_state(cache, request_identifier="req-A", layer_index=7)
        assert state_again is state
        assert state_again.decode_step_count == 2
        assert torch.all(state_again.recurrent_state == 3.14)

    def test_different_layers_get_different_states(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Different layer indices produce independent state objects."""
        state_3 = _init_state(cache, request_identifier="req-A", layer_index=3)
        state_7 = _init_state(cache, request_identifier="req-A", layer_index=7)
        assert state_3 is not state_7
        assert state_3.layer_index == 3
        assert state_7.layer_index == 7


# ===========================================================================
# Tests: Recycled state reuse (same data_ptr, no reallocation)
# ===========================================================================


class TestRecycledStateReuse:
    """Verify that recycled state is reused rather than reallocated."""

    def test_recycled_state_is_claimed_not_reallocated(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """A recycled state is claimed for the new request, reusing memory."""
        # Initialize for first request
        state_1 = _init_state(cache, request_identifier="req-A", layer_index=5)
        original_ptr = state_1.recurrent_state.data_ptr()

        # Recycle the state (simulating request completion)
        state_1.recycle()

        # Initialize for a new request on the same layer
        state_2 = _init_state(cache, request_identifier="req-B", layer_index=5)

        # Same underlying memory reused
        assert state_2.recurrent_state.data_ptr() == original_ptr
        # New request identifier
        assert state_2.request_identifier == "req-B"
        # Not available for reuse anymore
        assert not state_2.is_available_for_reuse

    def test_recycled_state_is_zeroed(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Claimed recycled state has zeroed tensors (from the recycle call)."""
        state = _init_state(cache, request_identifier="req-A", layer_index=5)
        state.recurrent_state.fill_(99.0)
        state.recycle()

        # Re-initialize for new request
        state_reused = _init_state(cache, request_identifier="req-B", layer_index=5)
        assert torch.all(state_reused.recurrent_state == 0)

    def test_recycled_state_decode_count_reset(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Claimed recycled state has decode step count reset to zero."""
        state = _init_state(cache, request_identifier="req-A", layer_index=5)
        state.increment_decode_step()
        state.increment_decode_step()
        state.recycle()

        state_reused = _init_state(cache, request_identifier="req-B", layer_index=5)
        assert state_reused.decode_step_count == 0

    def test_shape_mismatch_creates_fresh_state(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """If recycled state has incompatible shape, a fresh state is created."""
        # Create state with standard dimensions
        state_1 = _init_state(cache, request_identifier="req-A", layer_index=5)
        original_ptr = state_1.recurrent_state.data_ptr()
        state_1.recycle()

        # Initialize with different dimensions (different num_heads)
        state_2 = initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-B",
            layer_index=5,
            batch_size=_BATCH_SIZE,
            num_heads=64,  # Different from _NUM_HEADS=32
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=torch.device("cpu"),
        )

        # Fresh allocation — different data_ptr
        assert state_2.recurrent_state.data_ptr() != original_ptr
        assert state_2.shape_metadata.num_heads == 64


# ===========================================================================
# Tests: Different requests get independent states
# ===========================================================================


class TestRequestIsolation:
    """Verify that different requests get independent state storage."""

    def test_different_requests_same_layer_get_independent_states(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Two different requests on the same layer get independent states.

        In practice, a layer only holds one state at a time (the active request).
        This test verifies that after one request completes and a new one starts,
        the new request gets its own state.
        """
        # First request
        state_a = _init_state(cache, request_identifier="req-A", layer_index=5)
        state_a.recurrent_state.fill_(1.0)

        # Complete first request
        state_a.recycle()

        # Second request
        state_b = _init_state(cache, request_identifier="req-B", layer_index=5)

        # State B is independent (zeroed from recycle)
        assert state_b.request_identifier == "req-B"
        assert torch.all(state_b.recurrent_state == 0)

    def test_different_requests_different_layers_independent(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Different requests on different layers are fully independent."""
        state_a = _init_state(cache, request_identifier="req-A", layer_index=3)
        state_b = _init_state(cache, request_identifier="req-A", layer_index=7)

        state_a.recurrent_state.fill_(1.0)
        state_b.recurrent_state.fill_(2.0)

        assert torch.all(state_a.recurrent_state == 1.0)
        assert torch.all(state_b.recurrent_state == 2.0)
        assert state_a.recurrent_state.data_ptr() != state_b.recurrent_state.data_ptr()

    def test_stale_request_state_is_replaced(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """If a layer has active state for a different request, it is replaced."""
        # Initialize for request A
        state_a = _init_state(cache, request_identifier="req-A", layer_index=5)
        state_a.recurrent_state.fill_(42.0)

        # Initialize for request B on the same layer (without recycling A first)
        # This handles the edge case where a request was not properly cleaned up
        state_b = _init_state(cache, request_identifier="req-B", layer_index=5)

        # State B is fresh (not carrying A's data)
        assert state_b.request_identifier == "req-B"
        assert torch.all(state_b.recurrent_state == 0)
        assert state_b is not state_a


# ===========================================================================
# Tests: Registration in cache
# ===========================================================================


class TestCacheRegistration:
    """Verify that initialized state is properly registered in the cache."""

    def test_state_registered_in_cache(self, cache: "GatedDeltaNetCache") -> None:
        """After initialization, the state is retrievable from the cache."""
        state = _init_state(cache, layer_index=5)
        assert cache.has_gated_deltanet_state(5)
        assert cache.get_gated_deltanet_state(5) is state

    def test_multiple_layers_registered(self, cache: "GatedDeltaNetCache") -> None:
        """Multiple layers can be initialized and all are registered."""
        for layer_idx in [1, 3, 5, 7, 9]:
            _init_state(cache, layer_index=layer_idx)

        assert cache.gated_deltanet_state_count == 5
        assert cache.gated_deltanet_layer_indices == [1, 3, 5, 7, 9]

    def test_state_request_identifier_matches(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """The registered state has the correct request identifier."""
        state = _init_state(cache, request_identifier="my-request-xyz", layer_index=5)
        assert state.request_identifier == "my-request-xyz"

    def test_state_layer_index_matches(self, cache: "GatedDeltaNetCache") -> None:
        """The registered state has the correct layer index."""
        state = _init_state(cache, layer_index=12)
        assert state.layer_index == 12
