"""Preservation property tests for non-linear-attention paths.

These tests verify that full attention layers, MLP sharding, and utility functions
produce correct output on UNFIXED code. These paths are NOT affected by the
linear attention cache bug and must remain unchanged after the fix.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

Property 2: Preservation - Non-Linear-Attention Path Unchanged

For any forward call that does NOT flow through a native linear attention layer
(full attention layers, non-hybrid models, MLP blocks), the fixed code SHALL
produce exactly the same output as the original code, preserving all existing
tensor-parallel sharding, all-reduce synchronization, and KV cache behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from unittest.mock import patch

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
    _LINEAR_ATTN_PATTERN,
)


# --- Strategies ---


@st.composite
def compatible_dimensions(draw: st.DrawFn) -> dict:
    """Generate dimensions compatible with TPShardConfig constraints.

    Produces hidden_size, num_attention_heads, head_dim, intermediate_size,
    num_key_value_heads that are all divisible by world_size=2.
    """
    # Keep dimensions small for fast tests
    num_attention_heads = draw(st.sampled_from([4, 8]))
    num_key_value_heads = draw(st.sampled_from([2, 4]))
    # Ensure num_kv_heads divides num_attention_heads
    assume(num_attention_heads % num_key_value_heads == 0)
    # Ensure both divisible by world_size=2
    assume(num_attention_heads % 2 == 0)
    assume(num_key_value_heads % 2 == 0)

    head_dim = draw(st.sampled_from([16, 32]))
    hidden_size = num_attention_heads * head_dim
    intermediate_size = draw(st.sampled_from([hidden_size * 2, hidden_size * 4]))
    # Ensure intermediate_size divisible by world_size=2
    assume(intermediate_size % 2 == 0)

    batch_size = draw(st.integers(min_value=1, max_value=4))
    seq_len = draw(st.integers(min_value=1, max_value=16))

    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


# --- Helpers ---


def build_full_attn_state_dict(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    """Build a state_dict with full attention layers (no linear_attn keys).

    This simulates a non-hybrid model (like Phi-4 or Qwen2.5) where all
    layers use standard self-attention.
    """
    state_dict: dict[str, torch.Tensor] = {}
    prefix = "model.language_model"

    # Embedding
    vocab_size = 100
    state_dict[f"{prefix}.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)

    for idx in range(num_layers):
        lp = f"{prefix}.layers.{idx}"

        # Input layernorm
        state_dict[f"{lp}.input_layernorm.weight"] = torch.ones(hidden_size)

        # Self-attention QKV (column-parallel)
        state_dict[f"{lp}.self_attn.q_proj.weight"] = torch.randn(
            num_attention_heads * head_dim, hidden_size
        )
        state_dict[f"{lp}.self_attn.k_proj.weight"] = torch.randn(
            num_key_value_heads * head_dim, hidden_size
        )
        state_dict[f"{lp}.self_attn.v_proj.weight"] = torch.randn(
            num_key_value_heads * head_dim, hidden_size
        )
        # Output projection (row-parallel)
        state_dict[f"{lp}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, num_attention_heads * head_dim
        )

        # Post-attention layernorm
        state_dict[f"{lp}.post_attention_layernorm.weight"] = torch.ones(hidden_size)

        # MLP (gate/up column-parallel, down row-parallel)
        state_dict[f"{lp}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{lp}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{lp}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )

    # Final norm
    state_dict[f"{prefix}.norm.weight"] = torch.ones(hidden_size)

    # LM head
    state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    return state_dict


def build_hybrid_state_dict(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    """Build a state_dict with hybrid layers (linear_attn + full_attn).

    Simulates Qwen3.5/3.6 where 75% of layers are linear attention and
    25% are full attention (every 4th layer).
    """
    state_dict: dict[str, torch.Tensor] = {}
    prefix = "model.language_model"

    vocab_size = 100
    state_dict[f"{prefix}.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)

    for idx in range(num_layers):
        lp = f"{prefix}.layers.{idx}"
        is_full_attn = (idx % 4 == 3)

        # Input layernorm
        state_dict[f"{lp}.input_layernorm.weight"] = torch.ones(hidden_size)

        if is_full_attn:
            # Full attention layer
            state_dict[f"{lp}.self_attn.q_proj.weight"] = torch.randn(
                num_attention_heads * head_dim, hidden_size
            )
            state_dict[f"{lp}.self_attn.k_proj.weight"] = torch.randn(
                num_key_value_heads * head_dim, hidden_size
            )
            state_dict[f"{lp}.self_attn.v_proj.weight"] = torch.randn(
                num_key_value_heads * head_dim, hidden_size
            )
            state_dict[f"{lp}.self_attn.o_proj.weight"] = torch.randn(
                hidden_size, num_attention_heads * head_dim
            )
        else:
            # Linear attention layer (Gated DeltaNet)
            qkv_dim = hidden_size
            state_dict[f"{lp}.linear_attn.in_proj_qkv.weight"] = torch.randn(
                qkv_dim, hidden_size
            )
            state_dict[f"{lp}.linear_attn.in_proj_a.weight"] = torch.randn(
                32, hidden_size
            )
            state_dict[f"{lp}.linear_attn.in_proj_b.weight"] = torch.randn(
                32, hidden_size
            )
            state_dict[f"{lp}.linear_attn.in_proj_z.weight"] = torch.randn(
                hidden_size // 2, hidden_size
            )
            state_dict[f"{lp}.linear_attn.conv1d.weight"] = torch.randn(
                qkv_dim, 1, 4
            )
            state_dict[f"{lp}.linear_attn.out_proj.weight"] = torch.randn(
                hidden_size, hidden_size // 2
            )
            state_dict[f"{lp}.linear_attn.norm.weight"] = torch.ones(
                hidden_size // 2
            )
            state_dict[f"{lp}.linear_attn.A_log"] = torch.randn(32)
            state_dict[f"{lp}.linear_attn.dt_bias"] = torch.randn(32)

        # Post-attention layernorm
        state_dict[f"{lp}.post_attention_layernorm.weight"] = torch.ones(hidden_size)

        # MLP (always present in all layers)
        state_dict[f"{lp}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{lp}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{lp}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )

    # Final norm
    state_dict[f"{prefix}.norm.weight"] = torch.ones(hidden_size)

    # LM head
    state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    return state_dict


def create_tps_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> TensorParallelShard:
    """Create a TensorParallelShard from a raw state_dict (no model object).

    This bypasses _extract_native_linear_attn_layers since we pass a dict
    directly, which is the non-hybrid model path.
    """
    config = TPShardConfig(
        rank=0,
        world_size=2,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
    )
    # Patch _all_reduce to be a no-op for single-rank testing
    with patch.object(TensorParallelShard, '_all_reduce', side_effect=lambda t, **kw: t):
        tps = TensorParallelShard(state_dict, config, device="cpu")
    # Replace _all_reduce with identity for forward calls too
    tps._all_reduce = lambda t, **kw: t  # type: ignore[assignment]
    return tps


# --- Property-Based Tests ---


@given(data=st.data())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
def test_full_attention_output_shape_and_finite(data: st.DataObject) -> None:
    """Property 2: Full attention forward produces correct shape and finite values.

    **Validates: Requirements 3.1, 3.2**

    For all valid inputs to full attention layers, output shape is
    (batch, seq_len, hidden_size) and values are finite (no NaN/Inf).
    """
    dims = data.draw(compatible_dimensions())
    hidden_size = dims["hidden_size"]
    num_attention_heads = dims["num_attention_heads"]
    num_key_value_heads = dims["num_key_value_heads"]
    head_dim = dims["head_dim"]
    intermediate_size = dims["intermediate_size"]
    batch_size = dims["batch_size"]
    seq_len = dims["seq_len"]

    # Build a non-hybrid model state_dict (all full attention)
    num_layers = 2
    state_dict = build_full_attn_state_dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    tps = create_tps_from_state_dict(
        state_dict, hidden_size, num_attention_heads,
        num_key_value_heads, head_dim, intermediate_size,
    )

    # Create input token IDs (within vocab range)
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits, kv_cache = tps.forward(input_ids)

    # Property: output shape is (batch, seq_len, vocab_size=100)
    assert logits.shape == (batch_size, seq_len, 100), (
        f"Expected logits shape ({batch_size}, {seq_len}, 100), got {logits.shape}"
    )

    # Property: all values are finite (no NaN/Inf)
    assert torch.isfinite(logits).all(), (
        f"Logits contain non-finite values: "
        f"NaN count={torch.isnan(logits).sum().item()}, "
        f"Inf count={torch.isinf(logits).sum().item()}"
    )

    # Property: KV cache has correct structure
    assert len(kv_cache) == num_layers
    for layer_idx, kv in enumerate(kv_cache):
        assert kv is not None, f"Layer {layer_idx} KV cache should not be None for full attention"
        k, v = kv
        kv_heads_per_rank = num_key_value_heads // 2  # world_size=2
        assert k.shape == (batch_size, kv_heads_per_rank, seq_len, head_dim), (
            f"Layer {layer_idx} K shape mismatch: expected "
            f"({batch_size}, {kv_heads_per_rank}, {seq_len}, {head_dim}), got {k.shape}"
        )
        assert v.shape == (batch_size, kv_heads_per_rank, seq_len, head_dim), (
            f"Layer {layer_idx} V shape mismatch: expected "
            f"({batch_size}, {kv_heads_per_rank}, {seq_len}, {head_dim}), got {v.shape}"
        )


@given(data=st.data())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
def test_mlp_output_shape_and_finite(data: st.DataObject) -> None:
    """Property 2: MLP forward produces correct shape and finite values.

    **Validates: Requirements 3.3**

    For all valid inputs to MLP blocks, output shape is
    (batch, seq_len, hidden_size) and values are finite.
    """
    dims = data.draw(compatible_dimensions())
    hidden_size = dims["hidden_size"]
    num_attention_heads = dims["num_attention_heads"]
    num_key_value_heads = dims["num_key_value_heads"]
    head_dim = dims["head_dim"]
    intermediate_size = dims["intermediate_size"]
    batch_size = dims["batch_size"]
    seq_len = dims["seq_len"]

    # Build state_dict
    num_layers = 1
    state_dict = build_full_attn_state_dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    tps = create_tps_from_state_dict(
        state_dict, hidden_size, num_attention_heads,
        num_key_value_heads, head_dim, intermediate_size,
    )

    # Test MLP path directly: gate/up (column-parallel) → SiLU → down (row-parallel)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    prefix = f"{tps._layer_prefix}.layers.0"
    gate_weight = tps._get_weight(f"{prefix}.mlp.gate_proj.weight")
    up_weight = tps._get_weight(f"{prefix}.mlp.up_proj.weight")
    down_weight = tps._get_weight(f"{prefix}.mlp.down_proj.weight")

    with torch.no_grad():
        gate = tps._column_parallel_linear(hidden_states, gate_weight)
        up = tps._column_parallel_linear(hidden_states, up_weight)
        mlp_hidden = torch.nn.functional.silu(gate) * up
        mlp_output = tps._row_parallel_linear(mlp_hidden, down_weight, layer_index=0)

    # Property: output shape is (batch, seq_len, hidden_size)
    assert mlp_output.shape == (batch_size, seq_len, hidden_size), (
        f"Expected MLP output shape ({batch_size}, {seq_len}, {hidden_size}), "
        f"got {mlp_output.shape}"
    )

    # Property: all values are finite
    assert torch.isfinite(mlp_output).all(), (
        f"MLP output contains non-finite values: "
        f"NaN count={torch.isnan(mlp_output).sum().item()}, "
        f"Inf count={torch.isinf(mlp_output).sum().item()}"
    )


@given(
    param_name=st.sampled_from([
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.5.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.12.linear_attn.conv1d.weight",
        "model.language_model.layers.27.linear_attn.out_proj.weight",
    ])
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_is_redundant_identifies_linear_attn_weights(param_name: str) -> None:
    """Property 2: _is_redundant() correctly identifies linear_attn weights as redundant.

    **Validates: Requirements 3.5**

    For all parameter names matching _LINEAR_ATTN_PATTERN, _is_redundant()
    returns True, ensuring linear attention weights are kept redundant
    (not sharded) on all ranks.
    """
    # Verify the pattern is present in the param name
    assert _LINEAR_ATTN_PATTERN in param_name, (
        f"Test setup error: '{param_name}' does not contain '{_LINEAR_ATTN_PATTERN}'"
    )

    # Property: _is_redundant returns True for all linear_attn parameters
    result = TensorParallelShard._is_redundant(param_name)
    assert result is True, (
        f"_is_redundant('{param_name}') returned False, but linear_attn weights "
        f"must be kept redundant on all ranks per _LINEAR_ATTN_PATTERN='{_LINEAR_ATTN_PATTERN}'"
    )


@given(
    param_name=st.sampled_from([
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ])
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
def test_is_redundant_does_not_match_non_linear_attn(param_name: str) -> None:
    """Property 2: _is_redundant() does NOT mark attention/MLP weights as redundant.

    **Validates: Requirements 3.2, 3.3**

    For parameter names that are attention or MLP weights (not linear_attn),
    _is_redundant() returns False, ensuring they are properly sharded.
    """
    # Property: _is_redundant returns False for shardable parameters
    result = TensorParallelShard._is_redundant(param_name)
    assert result is False, (
        f"_is_redundant('{param_name}') returned True, but attention/MLP weights "
        f"should be sharded across ranks, not kept redundant"
    )


@given(
    num_layers=st.integers(min_value=4, max_value=8),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
def test_detect_layer_types_classifies_correctly(num_layers: int) -> None:
    """Property 2: _detect_layer_types() correctly classifies layers from state_dict keys.

    **Validates: Requirements 3.1, 3.2, 3.4**

    For all layer indices, _detect_layer_types() returns "linear_attention" iff
    linear_attn.in_proj_qkv.weight key exists for that layer, and "full_attention"
    iff self_attn.q_proj.weight key exists.
    """
    # Use fixed dimensions for this test
    hidden_size = 64
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 16
    intermediate_size = 128

    # Build hybrid state_dict (75% linear, 25% full)
    state_dict = build_hybrid_state_dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    tps = create_tps_from_state_dict(
        state_dict, hidden_size, num_attention_heads,
        num_key_value_heads, head_dim, intermediate_size,
    )

    layer_types = tps._layer_types

    # Property: correct number of layers detected
    assert len(layer_types) == num_layers, (
        f"Expected {num_layers} layer types, got {len(layer_types)}"
    )

    # Property: each layer type matches the presence of linear_attn keys
    for idx in range(num_layers):
        linear_key = f"{tps._layer_prefix}.layers.{idx}.linear_attn.in_proj_qkv.weight"
        full_key = f"{tps._layer_prefix}.layers.{idx}.self_attn.q_proj.weight"

        has_linear = linear_key in tps.sharded_state_dict
        has_full = full_key in tps.sharded_state_dict

        if has_linear:
            assert layer_types[idx] == "linear_attention", (
                f"Layer {idx} has linear_attn.in_proj_qkv.weight in state_dict "
                f"but _detect_layer_types() returned '{layer_types[idx]}' instead of 'linear_attention'"
            )
        elif has_full:
            assert layer_types[idx] == "full_attention", (
                f"Layer {idx} has self_attn.q_proj.weight in state_dict "
                f"but _detect_layer_types() returned '{layer_types[idx]}' instead of 'full_attention'"
            )


@given(data=st.data())
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
def test_full_attention_deterministic(data: st.DataObject) -> None:
    """Property 2: Full attention forward is deterministic for given input.

    **Validates: Requirements 3.1, 3.2**

    For the same input, the full attention path produces identical output
    across multiple calls (no randomness in inference mode).
    """
    dims = data.draw(compatible_dimensions())
    hidden_size = dims["hidden_size"]
    num_attention_heads = dims["num_attention_heads"]
    num_key_value_heads = dims["num_key_value_heads"]
    head_dim = dims["head_dim"]
    intermediate_size = dims["intermediate_size"]
    batch_size = dims["batch_size"]
    seq_len = dims["seq_len"]

    num_layers = 1
    state_dict = build_full_attn_state_dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    tps = create_tps_from_state_dict(
        state_dict, hidden_size, num_attention_heads,
        num_key_value_heads, head_dim, intermediate_size,
    )

    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    with torch.no_grad():
        logits_1, _ = tps.forward(input_ids)
        logits_2, _ = tps.forward(input_ids)

    # Property: deterministic output
    assert torch.allclose(logits_1, logits_2, atol=1e-6), (
        f"Full attention forward is not deterministic: "
        f"max diff = {(logits_1 - logits_2).abs().max().item()}"
    )
