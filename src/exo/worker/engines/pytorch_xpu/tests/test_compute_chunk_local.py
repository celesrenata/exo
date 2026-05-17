"""Tests for compute_chunk_local — chunk-level intra-chunk computation.

Verifies:
- Output shape matches (B, C, H, d_v)
- ChunkTransform has correct shapes and fp32 dtype
- Single-token chunk produces same output as the recurrent step
- Cumulative log-decay is the sum of per-token gates

**Validates: Requirements 7.1, 7.5, 7.6, 7.7**
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
_CHUNKED_PREFILL_PATH = _THIS_DIR.parent / "chunked_prefill.py"
_GATED_DELTANET_PATH = _THIS_DIR.parent / "gated_deltanet.py"


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


_chunked_mod = _load_module("chunked_prefill_compute_isolated", _CHUNKED_PREFILL_PATH)
_deltanet_mod = _load_module("gated_deltanet_compute_isolated", _GATED_DELTANET_PATH)

compute_chunk_local = _chunked_mod.compute_chunk_local
ChunkOutput = _chunked_mod.ChunkOutput
ChunkTransform = _chunked_mod.ChunkTransform
validate_chunk_transform_shapes = _chunked_mod.validate_chunk_transform_shapes

_gated_deltanet_recurrent_step_impl = _deltanet_mod._gated_deltanet_recurrent_step_impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along the given dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _make_chunk_inputs(
    batch_size: int = 2,
    chunk_size: int = 8,
    num_heads: int = 4,
    key_dim: int = 16,
    value_dim: int = 32,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random chunk inputs with proper constraints.

    Returns (q, k, v, gate, beta) with:
    - q, k: L2-normalized
    - gate: negative log-space values (decay)
    - beta: values in (0, 1)
    """
    q = _l2_normalize(torch.randn(batch_size, chunk_size, num_heads, key_dim, dtype=dtype))
    k = _l2_normalize(torch.randn(batch_size, chunk_size, num_heads, key_dim, dtype=dtype))
    v = torch.randn(batch_size, chunk_size, num_heads, value_dim, dtype=dtype)
    # gate in log-space: negative values so exp(gate) in (0, 1]
    gate = -torch.rand(batch_size, chunk_size, num_heads, dtype=dtype).abs() * 0.5
    # beta in (0, 1) via sigmoid
    beta = torch.sigmoid(torch.randn(batch_size, chunk_size, num_heads, dtype=dtype))
    return q, k, v, gate, beta


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------


class TestComputeChunkLocalOutputShapes:
    """Verify output shapes match expected dimensions."""

    def test_output_activations_shape(self) -> None:
        batch_size, chunk_size, num_heads, key_dim, value_dim = 2, 8, 4, 16, 32
        q, k, v, gate, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)

        chunk_output, _ = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        assert chunk_output.activations.shape == (batch_size, chunk_size, num_heads, value_dim)

    def test_output_activations_shape_various_sizes(self) -> None:
        for batch_size in (1, 3):
            for chunk_size in (1, 4, 16):
                for num_heads in (1, 8):
                    for key_dim in (8, 32):
                        for value_dim in (8, 64):
                            q, k, v, gate, beta = _make_chunk_inputs(
                                batch_size, chunk_size, num_heads, key_dim, value_dim
                            )
                            chunk_output, _ = compute_chunk_local(
                                q_chunk=q, k_chunk=k, v_chunk=v,
                                gate_chunk=gate, beta_chunk=beta, chunk_index=0,
                            )
                            assert chunk_output.activations.shape == (
                                batch_size, chunk_size, num_heads, value_dim
                            )

    def test_chunk_output_metadata(self) -> None:
        q, k, v, gate, beta = _make_chunk_inputs(chunk_size=16)

        chunk_output, _ = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=5,
        )

        assert chunk_output.chunk_index == 5
        assert chunk_output.chunk_size == 16


# ---------------------------------------------------------------------------
# ChunkTransform shape and dtype tests
# ---------------------------------------------------------------------------


class TestComputeChunkLocalTransformShapes:
    """Verify ChunkTransform has correct shapes and fp32 dtype."""

    def test_transform_shapes(self) -> None:
        batch_size, chunk_size, num_heads, key_dim, value_dim = 2, 8, 4, 16, 32
        q, k, v, gate, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        assert chunk_transform.cumulative_log_decay.shape == (batch_size, num_heads)
        assert chunk_transform.correction_keys.shape == (batch_size, num_heads, chunk_size, key_dim)
        assert chunk_transform.correction_core.shape == (batch_size, num_heads, chunk_size, chunk_size)
        assert chunk_transform.additive_term.shape == (batch_size, num_heads, key_dim, value_dim)

    def test_transform_all_fp32(self) -> None:
        q, k, v, gate, beta = _make_chunk_inputs()

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        assert chunk_transform.cumulative_log_decay.dtype == torch.float32
        assert chunk_transform.correction_keys.dtype == torch.float32
        assert chunk_transform.correction_core.dtype == torch.float32
        assert chunk_transform.additive_term.dtype == torch.float32

    def test_transform_passes_validation(self) -> None:
        q, k, v, gate, beta = _make_chunk_inputs()

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Should not raise
        validate_chunk_transform_shapes(chunk_transform)

    def test_transform_metadata_fields(self) -> None:
        batch_size, chunk_size, num_heads, key_dim, value_dim = 1, 4, 2, 8, 16
        q, k, v, gate, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        assert chunk_transform.chunk_size == chunk_size
        assert chunk_transform.num_heads == num_heads
        assert chunk_transform.key_dim == key_dim
        assert chunk_transform.value_dim == value_dim

    def test_transform_fp32_from_bf16_inputs(self) -> None:
        """Verify fp32 output even when inputs are bf16."""
        q, k, v, gate, beta = _make_chunk_inputs(dtype=torch.float32)
        # Convert to bf16 to simulate real usage
        q_bf16 = q.to(torch.bfloat16)
        k_bf16 = k.to(torch.bfloat16)
        v_bf16 = v.to(torch.bfloat16)
        gate_bf16 = gate.to(torch.bfloat16)
        beta_bf16 = beta.to(torch.bfloat16)

        chunk_output, chunk_transform = compute_chunk_local(
            q_chunk=q_bf16, k_chunk=k_bf16, v_chunk=v_bf16,
            gate_chunk=gate_bf16, beta_chunk=beta_bf16, chunk_index=0,
        )

        # All outputs must be fp32
        assert chunk_output.activations.dtype == torch.float32
        assert chunk_transform.cumulative_log_decay.dtype == torch.float32
        assert chunk_transform.correction_keys.dtype == torch.float32
        assert chunk_transform.correction_core.dtype == torch.float32
        assert chunk_transform.additive_term.dtype == torch.float32


# ---------------------------------------------------------------------------
# Single-token chunk equivalence with recurrent step
# ---------------------------------------------------------------------------


class TestSingleTokenChunkEquivalence:
    """Verify single-token chunk produces same output as the recurrent step."""

    def test_single_token_matches_recurrent_step(self) -> None:
        """A chunk of size 1 with zero initial state must match the recurrent step."""
        batch_size, num_heads, key_dim, value_dim = 2, 4, 16, 32
        chunk_size = 1

        q, k, v, gate, beta = _make_chunk_inputs(
            batch_size, chunk_size, num_heads, key_dim, value_dim
        )

        # Compute via compute_chunk_local
        chunk_output, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Compute via recurrent step with zero initial state
        # _gated_deltanet_recurrent_step_impl expects (B, H, d_k), (B, H, d_v), (B, H)
        q_t = q[:, 0, :, :]  # (B, H, d_k)
        k_t = k[:, 0, :, :]  # (B, H, d_k)
        v_t = v[:, 0, :, :]  # (B, H, d_v)
        g_t = gate[:, 0, :]  # (B, H)
        b_t = beta[:, 0, :]  # (B, H)

        zero_state = torch.zeros(batch_size, num_heads, key_dim, value_dim)
        recurrent_output, recurrent_state = _gated_deltanet_recurrent_step_impl(
            q_t, k_t, v_t, g_t, b_t, zero_state
        )

        # The chunk output should match the recurrent output
        # chunk_output.activations is (B, 1, H, d_v), recurrent_output is (B, H, d_v)
        chunk_out_squeezed = chunk_output.activations[:, 0, :, :]  # (B, H, d_v)

        torch.testing.assert_close(
            chunk_out_squeezed, recurrent_output.float(),
            rtol=1e-5, atol=1e-5,
        )

    def test_single_token_state_matches_recurrent_state(self) -> None:
        """The additive_term for a single-token chunk must match the recurrent state."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 16
        chunk_size = 1

        q, k, v, gate, beta = _make_chunk_inputs(
            batch_size, chunk_size, num_heads, key_dim, value_dim
        )

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Recurrent step with zero initial state
        q_t = q[:, 0, :, :]
        k_t = k[:, 0, :, :]
        v_t = v[:, 0, :, :]
        g_t = gate[:, 0, :]
        b_t = beta[:, 0, :]

        zero_state = torch.zeros(batch_size, num_heads, key_dim, value_dim)
        _, recurrent_state = _gated_deltanet_recurrent_step_impl(
            q_t, k_t, v_t, g_t, b_t, zero_state
        )

        # The additive_term is the local state after processing the chunk
        # starting from zero. For a single token, this is the state after one step.
        torch.testing.assert_close(
            chunk_transform.additive_term, recurrent_state.float(),
            rtol=1e-5, atol=1e-5,
        )

    def test_multi_token_chunk_matches_sequential_recurrence(self) -> None:
        """A multi-token chunk must produce the same outputs as sequential recurrence."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 16
        chunk_size = 4

        q, k, v, gate, beta = _make_chunk_inputs(
            batch_size, chunk_size, num_heads, key_dim, value_dim
        )

        # Compute via compute_chunk_local
        chunk_output, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Compute via sequential recurrence with zero initial state
        state = torch.zeros(batch_size, num_heads, key_dim, value_dim)
        sequential_outputs = []

        for t in range(chunk_size):
            q_t = q[:, t, :, :].float()
            k_t = k[:, t, :, :].float()
            v_t = v[:, t, :, :].float()
            g_t = gate[:, t, :].float()
            b_t = beta[:, t, :].float()

            out_t, state = _gated_deltanet_recurrent_step_impl(
                q_t, k_t, v_t, g_t, b_t, state
            )
            sequential_outputs.append(out_t)

        sequential_output = torch.stack(sequential_outputs, dim=1)  # (B, C, H, d_v)

        # Compare outputs
        torch.testing.assert_close(
            chunk_output.activations, sequential_output.float(),
            rtol=1e-5, atol=1e-5,
        )

        # Compare final state (additive_term = local state from zero)
        torch.testing.assert_close(
            chunk_transform.additive_term, state.float(),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Cumulative log-decay tests
# ---------------------------------------------------------------------------


class TestCumulativeLogDecay:
    """Verify cumulative log-decay is the sum of per-token gates."""

    def test_cumulative_log_decay_is_sum_of_gates(self) -> None:
        """The cumulative_log_decay in ChunkTransform must equal sum of all gates."""
        batch_size, chunk_size, num_heads, key_dim, value_dim = 2, 8, 4, 16, 32
        q, k, v, gate, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Expected: sum of all gates across the chunk dimension
        expected_total_decay = gate.float().sum(dim=1)  # (B, H)

        torch.testing.assert_close(
            chunk_transform.cumulative_log_decay, expected_total_decay,
            rtol=1e-5, atol=1e-5,
        )

    def test_single_token_cumulative_decay_equals_gate(self) -> None:
        """For a single-token chunk, cumulative_log_decay equals the single gate value."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 16
        chunk_size = 1

        q, k, v, gate, beta = _make_chunk_inputs(
            batch_size, chunk_size, num_heads, key_dim, value_dim
        )

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        expected = gate[:, 0, :].float()  # (B, H)
        torch.testing.assert_close(
            chunk_transform.cumulative_log_decay, expected,
            rtol=1e-5, atol=1e-5,
        )

    def test_cumulative_decay_is_negative(self) -> None:
        """Since gates are negative (log-space decay), cumulative decay must be negative."""
        batch_size, chunk_size, num_heads, key_dim, value_dim = 2, 8, 4, 16, 32
        q, k, v, gate, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # All gates are negative, so cumulative sum is negative
        assert (chunk_transform.cumulative_log_decay <= 0).all()

    def test_zero_gates_produce_zero_cumulative_decay(self) -> None:
        """Zero gates (no decay) produce zero cumulative log-decay."""
        batch_size, chunk_size, num_heads, key_dim, value_dim = 1, 4, 2, 8, 16
        q, k, v, _, beta = _make_chunk_inputs(batch_size, chunk_size, num_heads, key_dim, value_dim)
        gate = torch.zeros(batch_size, chunk_size, num_heads)

        _, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        expected = torch.zeros(batch_size, num_heads)
        torch.testing.assert_close(
            chunk_transform.cumulative_log_decay, expected,
            rtol=1e-7, atol=1e-7,
        )
