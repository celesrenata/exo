"""
Integration tests for GatedDeltaNet state optimization end-to-end lifecycle.

Tests the full pipeline: GatedDeltaNetCache + initialize_gated_deltanet_state +
gated_deltanet_decode_recurrent_step_optimized working together through a
complete request lifecycle.

Covers:
1. Compare optimized recurrent step to baseline across full lifecycle
2. Verify persistent state dtype is fp32 throughout lifecycle
3. Verify state storage reused across decode steps (same data_ptr)
4. Verify reset removes request state

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.7, 5.8, 5.10**
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
    "gated_deltanet_state_integration_test",
    _ENGINE_DIR / "gated_deltanet_state.py",
)
_cache_mod = _load_module(
    "gated_deltanet_cache_integration_test",
    _ENGINE_DIR / "gated_deltanet_cache.py",
)
_deltanet_mod = _load_module(
    "gated_deltanet_integration_test",
    _ENGINE_DIR / "gated_deltanet.py",
)

GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache
initialize_gated_deltanet_state = _state_mod.initialize_gated_deltanet_state
_gated_deltanet_recurrent_step_impl = _deltanet_mod._gated_deltanet_recurrent_step_impl
gated_deltanet_decode_recurrent_step_optimized = (
    _deltanet_mod.gated_deltanet_decode_recurrent_step_optimized
)


# ===========================================================================
# Test constants — Qwen3.5-4B dimensions
# ===========================================================================

_BATCH_SIZE = 1
_NUM_HEADS = 32
_KEY_DIM = 128
_VALUE_DIM = 128
_CONV_DIM = 256
_CONV_KERNEL_SIZE = 4
_NUM_LAYERS = 3  # Simulate a small multi-layer shard


# ===========================================================================
# Helpers
# ===========================================================================


class FakeDynamicCache:
    """Minimal DynamicCache mock for integration testing."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.key_cache[index], self.value_cache[index])

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(len(self.key_cache)):
            yield (self.key_cache[i], self.value_cache[i])


def _make_random_inputs(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random bf16 inputs for a single decode step."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    v = torch.randn(_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM, generator=gen, dtype=torch.bfloat16)
    gate = torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(
        torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16)
    )
    return q, k, v, gate, beta


def _init_state_for_layer(
    cache: "GatedDeltaNetCache",
    request_identifier: str,
    layer_index: int,
) -> "GatedDeltaNetPersistentState":
    """Initialize persistent state for a given layer via the full initialization path."""
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
        device=torch.device("cpu"),
    )


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestFullLifecycleOptimizedMatchesBaseline:
    """End-to-end: create cache → init state → decode → verify output matches baseline."""

    def test_multi_layer_multi_step_matches_baseline(self) -> None:
        """Full lifecycle across multiple layers and decode steps matches baseline.

        Simulates a real decode loop: initialize state for each layer, run
        multiple decode steps through all layers, and verify outputs match
        the baseline implementation at every step.
        """
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # Initialize state for all layers
        states = []
        for layer_idx in range(_NUM_LAYERS):
            state = _init_state_for_layer(cache, "request-lifecycle-001", layer_idx)
            states.append(state)

        # Baseline states (independent fp32 tensors)
        baseline_states = [
            torch.zeros(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32)
            for _ in range(_NUM_LAYERS)
        ]

        # Run 8 decode steps across all layers
        num_steps = 8
        for step in range(num_steps):
            for layer_idx in range(_NUM_LAYERS):
                q, k, v, gate, beta = _make_random_inputs(seed=step * 100 + layer_idx * 7)

                # Baseline
                output_baseline, baseline_states[layer_idx] = (
                    _gated_deltanet_recurrent_step_impl(
                        q, k, v, gate, beta, baseline_states[layer_idx]
                    )
                )

                # Optimized
                output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, states[layer_idx]
                )

                # Outputs match
                torch.testing.assert_close(
                    output_optimized, output_baseline, rtol=1e-3, atol=1e-3
                )

        # Final states match across all layers
        for layer_idx in range(_NUM_LAYERS):
            torch.testing.assert_close(
                states[layer_idx].recurrent_state,
                baseline_states[layer_idx],
                rtol=1e-3,
                atol=1e-3,
            )


class TestPersistentStateDtypeFp32ThroughLifecycle:
    """Verify persistent state dtype remains fp32 throughout the full lifecycle."""

    def test_dtype_fp32_after_init_decode_reset_reinit(self) -> None:
        """State dtype is fp32 at every lifecycle stage: init, decode, reset, reinit."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # Phase 1: Initialize
        state = _init_state_for_layer(cache, "request-dtype-001", layer_index=0)
        assert state.recurrent_state.dtype == torch.float32

        # Phase 2: Run decode steps
        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state
            )
            assert state.recurrent_state.dtype == torch.float32

        # Phase 3: Reset
        state.reset()
        assert state.recurrent_state.dtype == torch.float32

        # Phase 4: Recycle and reinitialize for new request
        state.recycle()
        state_reused = _init_state_for_layer(cache, "request-dtype-002", layer_index=0)
        assert state_reused.recurrent_state.dtype == torch.float32

        # Phase 5: Decode again after reinit
        for step in range(3):
            q, k, v, gate, beta = _make_random_inputs(seed=step + 50)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state_reused
            )
            assert state_reused.recurrent_state.dtype == torch.float32


class TestStateStorageReusedAcrossDecodeSteps:
    """Verify state storage (data_ptr) is reused across decode steps — no reallocation."""

    def test_data_ptr_stable_across_decode_steps(self) -> None:
        """The recurrent state data_ptr does not change across multiple decode steps."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        state = _init_state_for_layer(cache, "request-reuse-001", layer_index=0)

        original_ptr = state.recurrent_state.data_ptr()
        output_ptr = state.output_buffer_bf16.data_ptr()

        for step in range(10):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            output = gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state
            )
            # State storage unchanged
            assert state.recurrent_state.data_ptr() == original_ptr
            # Output buffer reused
            assert output.data_ptr() == output_ptr

    def test_data_ptr_stable_across_reinit_via_recycle(self) -> None:
        """After recycle + reinit, the same memory is reused (no reallocation)."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # First request
        state_1 = _init_state_for_layer(cache, "request-A", layer_index=0)
        original_ptr = state_1.recurrent_state.data_ptr()

        # Run some decode steps
        for step in range(3):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(q, k, v, gate, beta, state_1)

        # Recycle (request completed)
        state_1.recycle()

        # Second request reuses the same storage
        state_2 = _init_state_for_layer(cache, "request-B", layer_index=0)
        assert state_2.recurrent_state.data_ptr() == original_ptr

        # Decode with new request still uses same storage
        for step in range(3):
            q, k, v, gate, beta = _make_random_inputs(seed=step + 100)
            gated_deltanet_decode_recurrent_step_optimized(q, k, v, gate, beta, state_2)
            assert state_2.recurrent_state.data_ptr() == original_ptr


class TestResetRemovesRequestState:
    """Verify reset and recycle properly remove request state."""

    def test_reset_zeros_state_after_decode(self) -> None:
        """After decode steps, reset zeros all state tensors."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        state = _init_state_for_layer(cache, "request-reset-001", layer_index=0)

        # Run decode steps to accumulate state
        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(q, k, v, gate, beta, state)

        # State is non-zero
        assert not torch.all(state.recurrent_state == 0)
        assert state.decode_step_count == 5

        # Reset
        state.reset()

        # State is zeroed
        assert torch.all(state.recurrent_state == 0)
        assert torch.all(state.conv_state == 0)
        assert torch.all(state.output_buffer_fp32 == 0)
        assert torch.all(state.output_buffer_bf16 == 0)
        assert state.decode_step_count == 0

    def test_recycle_clears_ownership_and_zeros_state(self) -> None:
        """Recycle clears request ownership and zeros all state."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        state = _init_state_for_layer(cache, "request-recycle-001", layer_index=0)

        # Run decode steps
        for step in range(3):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(q, k, v, gate, beta, state)

        # Recycle
        state.recycle()

        # Ownership cleared
        assert state.request_identifier == ""
        assert state.is_available_for_reuse
        # State zeroed
        assert torch.all(state.recurrent_state == 0)
        assert state.decode_step_count == 0

    def test_cache_clear_removes_all_states(self) -> None:
        """Clearing the cache removes all registered GatedDeltaNet states."""
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # Initialize states for multiple layers
        for layer_idx in range(_NUM_LAYERS):
            _init_state_for_layer(cache, "request-clear-001", layer_idx)

        assert cache.gated_deltanet_state_count == _NUM_LAYERS

        # Clear all states
        cache.clear_gated_deltanet_states()

        assert cache.gated_deltanet_state_count == 0
        for layer_idx in range(_NUM_LAYERS):
            assert cache.get_gated_deltanet_state(layer_idx) is None


class TestFullRequestLifecycleEndToEnd:
    """Complete lifecycle: init → decode → reset → new request → decode → verify."""

    def test_two_request_lifecycle_with_state_reuse(self) -> None:
        """Two sequential requests reuse state storage and produce correct outputs.

        Lifecycle:
        1. Create cache
        2. Initialize state for request A
        3. Run decode steps for request A
        4. Verify output matches baseline for request A
        5. Recycle state (request A completed)
        6. Initialize state for request B (reuses storage)
        7. Run decode steps for request B
        8. Verify output matches baseline for request B (independent of A)
        """
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # --- Request A ---
        state_a = _init_state_for_layer(cache, "request-A", layer_index=0)
        ptr_a = state_a.recurrent_state.data_ptr()

        baseline_state_a = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )

        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step * 3)
            output_baseline, baseline_state_a = _gated_deltanet_recurrent_step_impl(
                q, k, v, gate, beta, baseline_state_a
            )
            output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state_a
            )
            torch.testing.assert_close(output_optimized, output_baseline, rtol=1e-3, atol=1e-3)

        # Request A completed — recycle
        state_a.recycle()

        # --- Request B ---
        state_b = _init_state_for_layer(cache, "request-B", layer_index=0)

        # Verify storage reuse
        assert state_b.recurrent_state.data_ptr() == ptr_a

        # Request B starts fresh (independent of A)
        baseline_state_b = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )

        for step in range(5):
            # Different seeds from request A to ensure independence
            q, k, v, gate, beta = _make_random_inputs(seed=step * 3 + 1000)
            output_baseline, baseline_state_b = _gated_deltanet_recurrent_step_impl(
                q, k, v, gate, beta, baseline_state_b
            )
            output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state_b
            )
            torch.testing.assert_close(output_optimized, output_baseline, rtol=1e-3, atol=1e-3)

        # Final state matches baseline for request B
        torch.testing.assert_close(
            state_b.recurrent_state, baseline_state_b, rtol=1e-3, atol=1e-3
        )
