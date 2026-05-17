"""Tests for state prefix propagation and output materialization in chunked GatedDeltaNet prefill.

Verifies:
- materialize_chunk_outputs produces correct outputs for a single chunk (matches sequential)
- chunked_gated_deltanet_prefill matches _gated_deltanet_prefill_sequential_impl within tolerance
- Final state matches sequential final state
- Works with sequences not divisible by chunk_size

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
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CHUNKED_PREFILL_PATH = _THIS_DIR.parent / "chunked_prefill.py"


def _load_chunked_prefill_module() -> types.ModuleType:
    """Load chunked_prefill.py directly from file, avoiding __init__.py."""
    module_name = "chunked_prefill_materialize_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _CHUNKED_PREFILL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_chunked_prefill_module()
ChunkOutput = _mod.ChunkOutput
compute_chunk_local = _mod.compute_chunk_local
compose_chunk_transforms = _mod.compose_chunk_transforms
materialize_chunk_outputs = _mod.materialize_chunk_outputs
chunked_gated_deltanet_prefill = _mod.chunked_gated_deltanet_prefill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sequential_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full sequential recurrence and return (outputs, final_state).

    Args:
        q: (B, T, H, d_k)
        k: (B, T, H, d_k)
        v: (B, T, H, d_v)
        gate: (B, T, H)
        beta: (B, T, H)
        initial_state: Optional (B, H, d_k, d_v), defaults to zero.

    Returns:
        outputs: (B, T, H, d_v)
        final_state: (B, H, d_k, d_v)
    """
    batch_size, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]

    q_fp32 = q.float()
    k_fp32 = k.float()
    v_fp32 = v.float()
    gate_fp32 = gate.float()
    beta_fp32 = beta.float()

    scale = key_dim ** -0.5
    q_fp32 = q_fp32 * scale

    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = torch.zeros(batch_size, num_heads, key_dim, value_dim, dtype=torch.float32)

    outputs = torch.zeros(batch_size, seq_len, num_heads, value_dim, dtype=torch.float32)

    for t in range(seq_len):
        q_t = q_fp32[:, t, :, :]
        k_t = k_fp32[:, t, :, :]
        v_t = v_fp32[:, t, :, :]
        g_t = gate_fp32[:, t, :]
        b_t = beta_fp32[:, t, :]

        # Decay
        decay = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        state = state * decay

        # Retrieve
        k_expanded = k_t.unsqueeze(-1)
        retrieved = (state * k_expanded).sum(dim=-2)

        # Delta correction
        beta_expanded = b_t.unsqueeze(-1)
        delta = beta_expanded * (v_t - retrieved)

        # Write
        delta_expanded = delta.unsqueeze(-2)
        state = state + k_expanded * delta_expanded

        # Output
        q_expanded = q_t.unsqueeze(-1)
        output_t = (state * q_expanded).sum(dim=-2)
        outputs[:, t, :, :] = output_t

    return outputs, state


def _generate_random_inputs(
    batch_size: int = 1,
    seq_len: int = 16,
    num_heads: int = 2,
    key_dim: int = 8,
    value_dim: int = 8,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random inputs for the GatedDeltaNet recurrence.

    Returns (q, k, v, gate, beta) with appropriate shapes and constraints.
    """
    gen = torch.Generator().manual_seed(seed)

    q = torch.randn(batch_size, seq_len, num_heads, key_dim, generator=gen, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # L2-normalize

    k = torch.randn(batch_size, seq_len, num_heads, key_dim, generator=gen, dtype=torch.float32)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # L2-normalize

    v = torch.randn(batch_size, seq_len, num_heads, value_dim, generator=gen, dtype=torch.float32)

    gate = -torch.rand(batch_size, seq_len, num_heads, generator=gen, dtype=torch.float32) * 0.5

    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, generator=gen, dtype=torch.float32))

    return q, k, v, gate, beta


# ---------------------------------------------------------------------------
# Tests: materialize_chunk_outputs — single chunk matches sequential
# ---------------------------------------------------------------------------


class TestMaterializeChunkOutputsSingleChunk:
    """Verify materialize_chunk_outputs produces correct outputs for a single chunk."""

    def test_single_chunk_matches_sequential(self) -> None:
        """A single chunk with zero incoming state matches sequential exactly."""
        batch_size, seq_len, num_heads, key_dim, value_dim = 1, 8, 2, 8, 8
        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=42,
        )

        # Sequential reference
        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        # Chunked: single chunk covering full sequence
        chunk_out, chunk_transform = compute_chunk_local(
            q_chunk=q, k_chunk=k, v_chunk=v,
            gate_chunk=gate, beta_chunk=beta, chunk_index=0,
        )

        # Prefix states: [zero_state, final_state]
        prefix_states = compose_chunk_transforms([chunk_transform])

        # Materialize
        outputs, final_state = materialize_chunk_outputs(
            chunk_outputs=[chunk_out],
            prefix_states=prefix_states,
            q_chunks=[q],
            k_chunks=[k],
            gate_chunks=[gate],
            beta_chunks=[beta],
        )

        # With zero incoming state, inter-chunk contribution is zero
        # so outputs should match intra-chunk outputs exactly
        assert torch.allclose(outputs, seq_outputs, atol=1e-5), (
            f"Max output diff: {(outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(final_state, seq_final_state, atol=1e-5), (
            f"Max state diff: {(final_state - seq_final_state).abs().max().item()}"
        )

    def test_single_chunk_various_sizes(self) -> None:
        """Single chunk works for various sequence lengths."""
        for seq_len in (4, 8, 16, 32):
            q, k, v, gate, beta = _generate_random_inputs(
                batch_size=1, seq_len=seq_len,
                num_heads=2, key_dim=8, value_dim=8, seed=seq_len,
            )

            seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

            chunk_out, chunk_transform = compute_chunk_local(
                q_chunk=q, k_chunk=k, v_chunk=v,
                gate_chunk=gate, beta_chunk=beta, chunk_index=0,
            )
            prefix_states = compose_chunk_transforms([chunk_transform])

            outputs, final_state = materialize_chunk_outputs(
                chunk_outputs=[chunk_out],
                prefix_states=prefix_states,
                q_chunks=[q],
                k_chunks=[k],
                gate_chunks=[gate],
                beta_chunks=[beta],
            )

            assert torch.allclose(outputs, seq_outputs, atol=1e-5), (
                f"seq_len={seq_len}: max output diff = {(outputs - seq_outputs).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# Tests: chunked_gated_deltanet_prefill matches sequential
# ---------------------------------------------------------------------------


class TestChunkedPrefillMatchesSequential:
    """Verify chunked_gated_deltanet_prefill matches sequential implementation."""

    def test_exact_chunk_boundary(self) -> None:
        """Sequence length exactly divisible by chunk_size."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 12  # 3 chunks of 4

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=100,
        )

        # Sequential reference
        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        # Chunked
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # The chunked approach uses the WY approximation for prefix states,
        # which introduces some error. Tolerance matches Requirement 7.8.
        assert torch.allclose(chunked_outputs, seq_outputs, atol=5e-2), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=5e-2), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_not_divisible_by_chunk_size(self) -> None:
        """Sequence length NOT divisible by chunk_size."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 11  # 2 full chunks of 4 + 1 partial chunk of 3

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=200,
        )

        # Sequential reference
        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        # Chunked
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # WY approximation error accumulates across chunk boundaries.
        # With 3 chunks, tolerance matches existing composition tests (atol=0.1).
        assert torch.allclose(chunked_outputs, seq_outputs, atol=0.1), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.1), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_sequence_shorter_than_chunk_size(self) -> None:
        """Sequence shorter than chunk_size (single partial chunk)."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 64
        seq_len = 7  # Much shorter than chunk_size

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=300,
        )

        # Sequential reference
        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        # Chunked (single chunk, no composition needed)
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # Single chunk should match exactly (no WY approximation needed)
        assert torch.allclose(chunked_outputs, seq_outputs, atol=1e-5), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=1e-5), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_batched_inputs(self) -> None:
        """Batched inputs (B > 1) produce correct results."""
        batch_size, num_heads, key_dim, value_dim = 3, 2, 8, 8
        chunk_size = 4
        seq_len = 12

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=400,
        )

        # Sequential reference
        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        # Chunked
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        assert torch.allclose(chunked_outputs, seq_outputs, atol=5e-2), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=5e-2), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_larger_chunk_size_reduces_error(self) -> None:
        """Larger chunk sizes have fewer inter-chunk boundaries, reducing WY error."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        seq_len = 16

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=500,
        )

        seq_outputs, _ = _run_sequential_recurrence(q, k, v, gate, beta)

        # Small chunks (more boundaries, more WY error)
        small_outputs, _ = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=4,
        )

        # Large chunks (fewer boundaries, less WY error)
        large_outputs, _ = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=16,
        )

        small_error = (small_outputs - seq_outputs).abs().max().item()
        large_error = (large_outputs - seq_outputs).abs().max().item()

        # Single chunk (chunk_size >= seq_len) should be exact
        assert large_error < 1e-5, f"Single chunk error should be near-zero, got {large_error}"
        # Small chunks have more error (or equal if sequence is short)
        # This is a sanity check, not a strict inequality
        assert small_error >= large_error - 1e-6


# ---------------------------------------------------------------------------
# Tests: Final state matches sequential
# ---------------------------------------------------------------------------


class TestFinalStateMatchesSequential:
    """Verify the final recurrent state matches the sequential implementation."""

    def test_final_state_two_chunks(self) -> None:
        """Final state after two chunks matches sequential."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 8

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=600,
        )

        _, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        _, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # WY approximation error for 2 chunks (matches existing composition test tolerance)
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.1), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_final_state_many_chunks(self) -> None:
        """Final state after many chunks matches sequential."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 32  # 8 chunks

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=700,
        )

        _, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        _, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # More chunks means more WY approximation error accumulation
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.15), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_final_state_with_initial_state(self) -> None:
        """Final state with non-zero initial state matches sequential."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 8

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=800,
        )

        # Non-zero initial state
        gen = torch.Generator().manual_seed(999)
        initial_state = torch.randn(
            batch_size, num_heads, key_dim, value_dim,
            generator=gen, dtype=torch.float32,
        ) * 0.1

        # Sequential with initial state
        seq_outputs, seq_final_state = _run_sequential_recurrence(
            q, k, v, gate, beta, initial_state=initial_state,
        )

        # Chunked with initial state
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta,
            chunk_size=chunk_size, initial_state=initial_state,
        )

        # WY approximation error with initial state propagation through 2 chunks
        assert torch.allclose(chunked_outputs, seq_outputs, atol=0.12), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.12), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Sequences not divisible by chunk_size
# ---------------------------------------------------------------------------


class TestPartialChunks:
    """Verify correct handling of sequences not divisible by chunk_size."""

    def test_seq_len_one_more_than_chunk(self) -> None:
        """seq_len = chunk_size + 1 produces correct results."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 5  # 1 full chunk + 1 partial chunk of 1

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=900,
        )

        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        assert chunked_outputs.shape == seq_outputs.shape
        assert torch.allclose(chunked_outputs, seq_outputs, atol=5e-2), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=5e-2), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_seq_len_one_less_than_chunk(self) -> None:
        """seq_len = chunk_size - 1 (single partial chunk)."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 8
        seq_len = 7

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=1000,
        )

        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # Single partial chunk — should be exact
        assert torch.allclose(chunked_outputs, seq_outputs, atol=1e-5), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=1e-5), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_prime_seq_len(self) -> None:
        """Prime sequence length (not divisible by any small chunk_size)."""
        batch_size, num_heads, key_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 13  # Prime number: chunks of [4, 4, 4, 1]

        q, k, v, gate, beta = _generate_random_inputs(
            batch_size=batch_size, seq_len=seq_len,
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, seed=1100,
        )

        seq_outputs, seq_final_state = _run_sequential_recurrence(q, k, v, gate, beta)

        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        assert chunked_outputs.shape == (batch_size, seq_len, num_heads, value_dim)
        assert torch.allclose(chunked_outputs, seq_outputs, atol=0.1), (
            f"Max output diff: {(chunked_outputs - seq_outputs).abs().max().item()}"
        )
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.1), (
            f"Max state diff: {(chunked_final_state - seq_final_state).abs().max().item()}"
        )

    def test_output_shape_correct(self) -> None:
        """Output shape is always (B, T, H, d_v) regardless of chunk boundaries."""
        for seq_len in (3, 7, 11, 15, 16, 17, 31, 33):
            q, k, v, gate, beta = _generate_random_inputs(
                batch_size=2, seq_len=seq_len,
                num_heads=4, key_dim=8, value_dim=16, seed=seq_len * 3,
            )

            outputs, final_state = chunked_gated_deltanet_prefill(
                q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=8,
            )

            assert outputs.shape == (2, seq_len, 4, 16), (
                f"seq_len={seq_len}: got shape {outputs.shape}"
            )
            assert final_state.shape == (2, 4, 8, 16), (
                f"seq_len={seq_len}: got state shape {final_state.shape}"
            )
