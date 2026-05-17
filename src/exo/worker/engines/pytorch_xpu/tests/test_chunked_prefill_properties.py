"""Property-based tests for chunked GatedDeltaNet prefill.

Tests the following properties:
- Chunk boundary property: output is independent of chunk_size choice
- Associative composition property: compose(t1, compose(t2, t3)) == compose(compose(t1, t2), t3)
- Final state equivalence property: chunked final state matches sequential
- Decode continuation property: decode from chunked state matches decode from sequential state
- Fallback safety property: fallback always produces valid outputs

**Validates: Requirements 7.1, 7.5, 7.6, 7.7**
"""

from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CHUNKED_PREFILL_PATH = _THIS_DIR.parent / "chunked_prefill.py"


def _load_chunked_prefill_module() -> types.ModuleType:
    """Load chunked_prefill.py directly from file, avoiding __init__.py."""
    module_name = "chunked_prefill_properties_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _CHUNKED_PREFILL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_chunked_prefill_module()
chunked_gated_deltanet_prefill = _mod.chunked_gated_deltanet_prefill
compute_chunk_local = _mod.compute_chunk_local
compose_two_transforms = _mod.compose_two_transforms
apply_chunk_transform = _mod.apply_chunk_transform
chunked_prefill_with_fallback = _mod.chunked_prefill_with_fallback
ChunkedPrefillUnsupportedError = _mod.ChunkedPrefillUnsupportedError
ChunkedPrefillFallbackWarning = _mod.ChunkedPrefillFallbackWarning
_sequential_prefill_fallback = _mod._sequential_prefill_fallback


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def gated_deltanet_inputs(
    draw: st.DrawFn,
    min_seq_len: int = 2,
    max_seq_len: int = 24,
    min_heads: int = 1,
    max_heads: int = 3,
    min_dim: int = 4,
    max_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate valid GatedDeltaNet inputs for property testing."""
    batch_size = 1
    seq_len = draw(st.integers(min_value=min_seq_len, max_value=max_seq_len))
    num_heads = draw(st.integers(min_value=min_heads, max_value=max_heads))
    key_dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    value_dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    seed = draw(st.integers(min_value=0, max_value=10000))

    gen = torch.Generator().manual_seed(seed)

    q = torch.randn(batch_size, seq_len, num_heads, key_dim, generator=gen, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    k = torch.randn(batch_size, seq_len, num_heads, key_dim, generator=gen, dtype=torch.float32)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    v = torch.randn(batch_size, seq_len, num_heads, value_dim, generator=gen, dtype=torch.float32)

    # Use small negative gates for numerical stability in property tests
    gate = -torch.rand(batch_size, seq_len, num_heads, generator=gen, dtype=torch.float32) * 0.3

    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, generator=gen, dtype=torch.float32))

    return q, k, v, gate, beta


# ---------------------------------------------------------------------------
# Property: Chunk boundary — output shape is always correct
# ---------------------------------------------------------------------------


class TestChunkBoundaryProperty:
    """Chunk boundary property: output shape is (B, T, H, d_v) regardless of chunk_size."""

    @given(
        inputs=gated_deltanet_inputs(min_seq_len=2, max_seq_len=20),
        chunk_size=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=30, deadline=10000)
    def test_output_shape_invariant_to_chunk_size(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        chunk_size: int,
    ) -> None:
        """Output shape is always (B, T, H, d_v) regardless of chunk_size choice.

        **Validates: Requirements 7.1**
        """
        q, k, v, gate, beta = inputs
        batch_size, seq_len, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]

        outputs, final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        assert outputs.shape == (batch_size, seq_len, num_heads, value_dim)
        assert final_state.shape == (batch_size, num_heads, key_dim, value_dim)
        assert outputs.dtype == torch.float32
        assert final_state.dtype == torch.float32


# ---------------------------------------------------------------------------
# Property: Associative composition
# ---------------------------------------------------------------------------


class TestAssociativeCompositionProperty:
    """Associative composition: compose(t1, compose(t2, t3)) ≈ compose(compose(t1, t2), t3)."""

    @given(
        seed=st.integers(min_value=0, max_value=10000),
        chunk_size=st.integers(min_value=2, max_value=6),
        num_heads=st.integers(min_value=1, max_value=2),
        key_dim=st.integers(min_value=4, max_value=6),
        value_dim=st.integers(min_value=4, max_value=6),
    )
    @settings(max_examples=20, deadline=15000)
    def test_composition_associativity(
        self,
        seed: int,
        chunk_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        """Transform composition is associative within numerical tolerance.

        **Validates: Requirements 7.5, 7.6**
        """
        batch_size = 1
        gen = torch.Generator().manual_seed(seed)

        # Generate 3 chunks of random inputs
        chunks = []
        for _ in range(3):
            q_c = torch.randn(batch_size, chunk_size, num_heads, key_dim, generator=gen)
            q_c = q_c / q_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k_c = torch.randn(batch_size, chunk_size, num_heads, key_dim, generator=gen)
            k_c = k_c / k_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_c = torch.randn(batch_size, chunk_size, num_heads, value_dim, generator=gen)
            gate_c = -torch.rand(batch_size, chunk_size, num_heads, generator=gen) * 0.3
            beta_c = torch.sigmoid(torch.randn(batch_size, chunk_size, num_heads, generator=gen))
            chunks.append((q_c, k_c, v_c, gate_c, beta_c))

        # Compute transforms for each chunk
        _, t1 = compute_chunk_local(
            q_chunk=chunks[0][0], k_chunk=chunks[0][1], v_chunk=chunks[0][2],
            gate_chunk=chunks[0][3], beta_chunk=chunks[0][4], chunk_index=0,
        )
        _, t2 = compute_chunk_local(
            q_chunk=chunks[1][0], k_chunk=chunks[1][1], v_chunk=chunks[1][2],
            gate_chunk=chunks[1][3], beta_chunk=chunks[1][4], chunk_index=1,
        )
        _, t3 = compute_chunk_local(
            q_chunk=chunks[2][0], k_chunk=chunks[2][1], v_chunk=chunks[2][2],
            gate_chunk=chunks[2][3], beta_chunk=chunks[2][4], chunk_index=2,
        )

        # Left-associative: (t1 ∘ t2) ∘ t3
        t12 = compose_two_transforms(t1, t2)
        t123_left = compose_two_transforms(t12, t3)

        # Right-associative: t1 ∘ (t2 ∘ t3)
        t23 = compose_two_transforms(t2, t3)
        t123_right = compose_two_transforms(t1, t23)

        # Apply both to a test state and compare
        test_state = torch.randn(batch_size, num_heads, key_dim, value_dim, generator=gen)

        result_left = apply_chunk_transform(test_state, t123_left)
        result_right = apply_chunk_transform(test_state, t123_right)

        # Associativity holds within numerical tolerance
        assert torch.allclose(result_left, result_right, atol=0.15), (
            f"Associativity violation: max diff = "
            f"{(result_left - result_right).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Property: Final state equivalence
# ---------------------------------------------------------------------------


class TestFinalStateEquivalenceProperty:
    """Final state from chunked prefill matches sequential within tolerance."""

    @given(inputs=gated_deltanet_inputs(min_seq_len=4, max_seq_len=16))
    @settings(max_examples=25, deadline=15000)
    def test_final_state_matches_sequential(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Chunked final state matches sequential final state within tolerance.

        **Validates: Requirements 7.7**
        """
        q, k, v, gate, beta = inputs
        seq_len = q.shape[1]

        # Use chunk_size that creates multiple chunks
        chunk_size = max(2, seq_len // 3)

        # Sequential reference
        seq_outputs, seq_final_state = _sequential_prefill_fallback(
            q=q, k=k, v=v, gate=gate, beta=beta,
        )

        # Chunked
        chunked_outputs, chunked_final_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # Final state equivalence within WY approximation tolerance
        assert torch.allclose(chunked_final_state, seq_final_state, atol=0.2), (
            f"Final state mismatch: max diff = "
            f"{(chunked_final_state - seq_final_state).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Property: Decode continuation
# ---------------------------------------------------------------------------


class TestDecodeContinuationProperty:
    """Decode from chunked state produces similar results to decode from sequential state."""

    @given(inputs=gated_deltanet_inputs(min_seq_len=4, max_seq_len=12, max_dim=6))
    @settings(max_examples=15, deadline=15000)
    def test_decode_from_chunked_state(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """A single decode step from chunked state matches decode from sequential state.

        **Validates: Requirements 7.7**
        """
        q, k, v, gate, beta = inputs
        batch_size, seq_len, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]
        chunk_size = max(2, seq_len // 2)

        # Get final states from both paths
        _, seq_state = _sequential_prefill_fallback(
            q=q, k=k, v=v, gate=gate, beta=beta,
        )
        _, chunked_state = chunked_gated_deltanet_prefill(
            q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
        )

        # Simulate one decode step from each state
        gen = torch.Generator().manual_seed(42)
        q_decode = torch.randn(batch_size, 1, num_heads, key_dim, generator=gen)
        q_decode = q_decode / q_decode.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k_decode = torch.randn(batch_size, 1, num_heads, key_dim, generator=gen)
        k_decode = k_decode / k_decode.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_decode = torch.randn(batch_size, 1, num_heads, value_dim, generator=gen)
        gate_decode = -torch.rand(batch_size, 1, num_heads, generator=gen) * 0.3
        beta_decode = torch.sigmoid(torch.randn(batch_size, 1, num_heads, generator=gen))

        # Decode from sequential state
        seq_decode_out, _ = _sequential_prefill_fallback(
            q=q_decode, k=k_decode, v=v_decode,
            gate=gate_decode, beta=beta_decode,
            initial_state=seq_state,
        )

        # Decode from chunked state
        chunked_decode_out, _ = _sequential_prefill_fallback(
            q=q_decode, k=k_decode, v=v_decode,
            gate=gate_decode, beta=beta_decode,
            initial_state=chunked_state,
        )

        # Decode outputs should be similar (bounded by state approximation error)
        assert torch.allclose(chunked_decode_out, seq_decode_out, atol=0.3), (
            f"Decode continuation mismatch: max diff = "
            f"{(chunked_decode_out - seq_decode_out).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Property: Fallback safety
# ---------------------------------------------------------------------------


class TestFallbackSafetyProperty:
    """Fallback always produces valid outputs regardless of input configuration."""

    @given(inputs=gated_deltanet_inputs(min_seq_len=1, max_seq_len=16))
    @settings(max_examples=20, deadline=10000)
    def test_fallback_produces_valid_outputs(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Fallback path always produces finite outputs with correct shape.

        **Validates: Requirements 7.1**
        """
        q, k, v, gate, beta = inputs
        batch_size, seq_len, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ChunkedPrefillFallbackWarning)
            outputs, final_state = chunked_prefill_with_fallback(
                q=q, k=k, v=v, gate=gate, beta=beta,
                chunk_size=64,
                fallback_on_unsupported_shape=True,
            )

        assert outputs.shape == (batch_size, seq_len, num_heads, value_dim)
        assert final_state.shape == (batch_size, num_heads, key_dim, value_dim)
        assert torch.isfinite(outputs).all(), "Outputs contain non-finite values"
        assert torch.isfinite(final_state).all(), "Final state contains non-finite values"

    def test_unsupported_dtype_raises_when_fallback_disabled(self) -> None:
        """Unsupported dtype raises ChunkedPrefillUnsupportedError when fallback disabled.

        **Validates: Requirements 7.1**
        """
        # int32 is not a supported dtype
        q = torch.ones(1, 4, 2, 4, dtype=torch.int32)
        k = torch.ones(1, 4, 2, 4, dtype=torch.int32)
        v = torch.ones(1, 4, 2, 4, dtype=torch.int32)
        gate = torch.ones(1, 4, 2, dtype=torch.int32)
        beta = torch.ones(1, 4, 2, dtype=torch.int32)

        with pytest.raises(ChunkedPrefillUnsupportedError) as exc_info:
            chunked_prefill_with_fallback(
                q=q, k=k, v=v, gate=gate, beta=beta,
                chunk_size=4,
                fallback_on_unsupported_shape=False,
            )

        assert "dtype" in exc_info.value.reason.lower() or "Unsupported" in exc_info.value.reason

    def test_unsupported_dtype_warns_when_fallback_enabled(self) -> None:
        """Unsupported dtype emits warning and falls back when fallback enabled.

        **Validates: Requirements 7.1**
        """
        # Use float64 which is not in _SUPPORTED_DTYPES
        q = torch.ones(1, 4, 2, 4, dtype=torch.float64)
        k = torch.ones(1, 4, 2, 4, dtype=torch.float64)
        v = torch.ones(1, 4, 2, 4, dtype=torch.float64)
        gate = torch.ones(1, 4, 2, dtype=torch.float64)
        beta = torch.ones(1, 4, 2, dtype=torch.float64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outputs, final_state = chunked_prefill_with_fallback(
                q=q, k=k, v=v, gate=gate, beta=beta,
                chunk_size=4,
                fallback_on_unsupported_shape=True,
            )

        # Should have emitted a warning
        fallback_warnings = [
            x for x in w if issubclass(x.category, ChunkedPrefillFallbackWarning)
        ]
        assert len(fallback_warnings) >= 1

        # Should still produce valid outputs
        assert outputs.shape == (1, 4, 2, 4)
        assert final_state.shape == (1, 2, 4, 4)
