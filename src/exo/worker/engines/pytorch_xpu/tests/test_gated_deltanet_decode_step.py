"""
Unit tests for gated_deltanet_decode_recurrent_step_optimized.

Compares the optimized in-place decode step against the baseline
_gated_deltanet_recurrent_step_impl, verifies in-place state mutation,
preallocated buffer reuse, and multi-step state accumulation.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.10**
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
_PARENT_DIR = _THIS_DIR.parent

_STATE_MODULE_PATH = _PARENT_DIR / "gated_deltanet_state.py"
_DELTANET_MODULE_PATH = _PARENT_DIR / "gated_deltanet.py"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_state_mod = _load_module("gated_deltanet_state_test_decode", _STATE_MODULE_PATH)
_deltanet_mod = _load_module("gated_deltanet_test_decode", _DELTANET_MODULE_PATH)

GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
_gated_deltanet_recurrent_step_impl = _deltanet_mod._gated_deltanet_recurrent_step_impl
gated_deltanet_decode_recurrent_step_optimized = (
    _deltanet_mod.gated_deltanet_decode_recurrent_step_optimized
)


# ===========================================================================
# Test fixtures
# ===========================================================================

# Qwen3.5-4B dimensions
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
        request_identifier="test-decode-001",
        layer_index=0,
        batch_size=_BATCH_SIZE,
        num_heads=_NUM_HEADS,
        key_dim=_KEY_DIM,
        value_dim=_VALUE_DIM,
        conv_dim=_CONV_DIM,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        device=torch.device("cpu"),
    )


def _make_random_inputs(
    batch_size: int = _BATCH_SIZE,
    num_heads: int = _NUM_HEADS,
    key_dim: int = _KEY_DIM,
    value_dim: int = _VALUE_DIM,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random bf16 inputs for a single decode step."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch_size, num_heads, key_dim, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, key_dim, generator=gen, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, value_dim, generator=gen, dtype=torch.bfloat16)
    gate = torch.randn(batch_size, num_heads, generator=gen, dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(
        torch.randn(batch_size, num_heads, generator=gen, dtype=torch.bfloat16)
    )
    return q, k, v, gate, beta


# ===========================================================================
# Tests: Output equivalence with baseline
# ===========================================================================


class TestOutputEquivalence:
    """Verify optimized function produces same output as baseline within tolerance."""

    def test_single_step_output_matches_baseline(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Optimized output matches baseline output within bf16 tolerance."""
        q, k, v, gate, beta = _make_random_inputs()

        # Baseline: uses fresh zero state
        state_baseline = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )
        output_baseline, _new_state_baseline = _gated_deltanet_recurrent_step_impl(
            q, k, v, gate, beta, state_baseline
        )

        # Optimized: uses persistent state (starts at zero)
        output_optimized = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )

        assert output_optimized.dtype == torch.bfloat16
        assert output_baseline.dtype == torch.bfloat16
        torch.testing.assert_close(
            output_optimized, output_baseline, rtol=1e-3, atol=1e-3
        )

    def test_single_step_state_matches_baseline(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Optimized state matches baseline state within tolerance."""
        q, k, v, gate, beta = _make_random_inputs()

        # Baseline
        state_baseline = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )
        _output_baseline, new_state_baseline = _gated_deltanet_recurrent_step_impl(
            q, k, v, gate, beta, state_baseline
        )

        # Optimized
        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )

        torch.testing.assert_close(
            persistent_state.recurrent_state, new_state_baseline, rtol=1e-3, atol=1e-3
        )

    def test_multi_step_output_matches_baseline(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Multiple decode steps produce equivalent outputs to baseline."""
        num_steps = 5
        state_baseline = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )

        for step in range(num_steps):
            q, k, v, gate, beta = _make_random_inputs(seed=step * 7 + 13)

            output_baseline, state_baseline = _gated_deltanet_recurrent_step_impl(
                q, k, v, gate, beta, state_baseline
            )

            output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state
            )

            torch.testing.assert_close(
                output_optimized, output_baseline, rtol=1e-3, atol=1e-3
            )

        # Final state should also match
        torch.testing.assert_close(
            persistent_state.recurrent_state, state_baseline, rtol=1e-3, atol=1e-3
        )

    def test_nonzero_initial_state_matches_baseline(self) -> None:
        """Optimized function works correctly with non-zero initial state."""
        gen = torch.Generator().manual_seed(99)
        initial_state = torch.randn(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM,
            generator=gen, dtype=torch.float32
        )

        # Create persistent state and set initial state
        ps = GatedDeltaNetPersistentState.create(
            request_identifier="test-nonzero",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=torch.device("cpu"),
        )
        ps.recurrent_state.copy_(initial_state)

        q, k, v, gate, beta = _make_random_inputs(seed=77)

        # Baseline with same initial state
        output_baseline, new_state_baseline = _gated_deltanet_recurrent_step_impl(
            q, k, v, gate, beta, initial_state.clone()
        )

        # Optimized
        output_optimized = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, ps
        )

        torch.testing.assert_close(
            output_optimized, output_baseline, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            ps.recurrent_state, new_state_baseline, rtol=1e-3, atol=1e-3
        )


# ===========================================================================
# Tests: In-place state modification
# ===========================================================================


class TestInPlaceStateModification:
    """Verify state is modified in place (same data_ptr before and after)."""

    def test_state_data_ptr_unchanged(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """The recurrent state tensor's data_ptr does not change after a step."""
        original_ptr = persistent_state.recurrent_state.data_ptr()

        q, k, v, gate, beta = _make_random_inputs()
        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )

        assert persistent_state.recurrent_state.data_ptr() == original_ptr

    def test_state_data_ptr_stable_across_multiple_steps(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """State data_ptr remains stable across multiple decode steps."""
        original_ptr = persistent_state.recurrent_state.data_ptr()

        for step in range(10):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state
            )
            assert persistent_state.recurrent_state.data_ptr() == original_ptr

    def test_state_is_actually_modified(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """State is not zero after a decode step with non-zero inputs."""
        q, k, v, gate, beta = _make_random_inputs()
        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )
        assert not torch.all(persistent_state.recurrent_state == 0)


# ===========================================================================
# Tests: Preallocated output buffer usage
# ===========================================================================


class TestPreallocatedOutputBuffer:
    """Verify output uses the preallocated bf16 buffer."""

    def test_output_uses_preallocated_bf16_buffer(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output tensor's data_ptr matches persistent_state.output_buffer_bf16."""
        q, k, v, gate, beta = _make_random_inputs()
        output = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )
        assert output.data_ptr() == persistent_state.output_buffer_bf16.data_ptr()

    def test_output_dtype_is_bf16(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output is returned in bf16 (model compute dtype)."""
        q, k, v, gate, beta = _make_random_inputs()
        output = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )
        assert output.dtype == torch.bfloat16

    def test_output_shape_correct(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Output has shape (B, H, d_v)."""
        q, k, v, gate, beta = _make_random_inputs()
        output = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )
        assert output.shape == (_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM)

    def test_fp32_buffer_also_written(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """The fp32 output buffer is also populated (intermediate result)."""
        q, k, v, gate, beta = _make_random_inputs()
        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state
        )
        # fp32 buffer should not be all zeros after a step with non-zero inputs
        assert not torch.all(persistent_state.output_buffer_fp32 == 0)


# ===========================================================================
# Tests: Multi-step state accumulation
# ===========================================================================


class TestMultiStepAccumulation:
    """Verify state accumulates correctly over multiple decode steps."""

    def test_state_accumulates_over_steps(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """State norm increases over multiple steps (state accumulates information)."""
        norms = []
        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step + 100)
            # Use small positive gates to avoid state decay dominating
            gate = torch.full((_BATCH_SIZE, _NUM_HEADS), -0.01, dtype=torch.bfloat16)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state
            )
            norms.append(persistent_state.recurrent_state.norm().item())

        # State should grow (not stay at zero) — at least the last norm > first
        assert norms[-1] > norms[0]

    def test_decode_step_count_increments(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Decode step counter increments with each call."""
        assert persistent_state.decode_step_count == 0

        for step in range(3):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state
            )

        assert persistent_state.decode_step_count == 3


# ===========================================================================
# Tests: No per-token dtype promotion of state
# ===========================================================================


class TestNoDtypePromotion:
    """Verify the state remains fp32 without per-token casting."""

    def test_state_dtype_remains_fp32(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Recurrent state dtype stays fp32 across multiple decode steps."""
        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state
            )
            assert persistent_state.recurrent_state.dtype == torch.float32

    def test_inputs_are_bf16(
        self, persistent_state: "GatedDeltaNetPersistentState"
    ) -> None:
        """Input tensors are bf16 (confirming no external promotion needed)."""
        q, k, v, gate, beta = _make_random_inputs()
        assert q.dtype == torch.bfloat16
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16
        assert gate.dtype == torch.bfloat16
        assert beta.dtype == torch.bfloat16
