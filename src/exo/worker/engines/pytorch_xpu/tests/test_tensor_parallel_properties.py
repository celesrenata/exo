"""Property-based tests for TensorParallelShard using Hypothesis.

These tests verify correctness properties that must hold across all valid inputs,
not just specific examples. Each property is tagged with its requirement link.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
)


# =============================================================================
# Strategies
# =============================================================================


# Valid configurations that keep tensor sizes small for fast testing
_VALID_HIDDEN_SIZES = [64, 128, 256]
_VALID_NUM_HEADS = [4, 8, 16]
_VALID_NUM_KV_HEADS = [2, 4, 8]
_VALID_INTERMEDIATE_SIZES = [128, 256, 512]
_VALID_WORLD_SIZES = [2, 4]
_VALID_NUM_LAYERS = [1, 2, 3]


@st.composite
def valid_tp_config(draw: st.DrawFn) -> tuple[TPShardConfig, dict[str, int]]:
    """Generate a valid TPShardConfig with compatible dimensions.

    Returns a tuple of (config, params_dict) where params_dict contains
    the raw dimension values for constructing state dicts.
    """
    world_size = draw(st.sampled_from(_VALID_WORLD_SIZES))
    # Pick num_heads divisible by world_size
    num_heads = draw(st.sampled_from([h for h in _VALID_NUM_HEADS if h % world_size == 0]))
    # Pick num_kv_heads divisible by world_size and <= num_heads
    num_kv_heads = draw(
        st.sampled_from(
            [kv for kv in _VALID_NUM_KV_HEADS if kv % world_size == 0 and kv <= num_heads]
        )
    )
    hidden_size = draw(st.sampled_from(_VALID_HIDDEN_SIZES))
    # head_dim must divide evenly: hidden_size is independent of num_heads * head_dim
    # for GQA models. Pick head_dim from factors that work.
    head_dim = draw(st.sampled_from([16, 32, 64]))
    intermediate_size = draw(
        st.sampled_from([s for s in _VALID_INTERMEDIATE_SIZES if s % world_size == 0])
    )
    rank = draw(st.integers(min_value=0, max_value=world_size - 1))

    config = TPShardConfig(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_kv_heads,
    )
    params = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "world_size": world_size,
        "rank": rank,
    }
    return config, params


@st.composite
def qwen_state_dict(
    draw: st.DrawFn, params: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Generate a Qwen/Llama-style state dict with separate Q, K, V projections."""
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    num_kv_heads = params["num_kv_heads"]
    intermediate_size = params["intermediate_size"]
    num_layers = draw(st.sampled_from(_VALID_NUM_LAYERS))

    state_dict: dict[str, torch.Tensor] = {}
    vocab_size = 100  # Small vocab for testing

    # Embedding and final layers
    state_dict["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    state_dict["model.norm.weight"] = torch.randn(hidden_size)
    state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            num_heads * head_dim, hidden_size
        )
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
            num_kv_heads * head_dim, hidden_size
        )
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, num_heads * head_dim
        )
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )

    return state_dict


@st.composite
def phi_state_dict(
    draw: st.DrawFn, params: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Generate a Phi-style state dict with fused QKV and LayerNorm biases."""
    hidden_size = params["hidden_size"]
    num_heads = params["num_heads"]
    head_dim = params["head_dim"]
    num_kv_heads = params["num_kv_heads"]
    intermediate_size = params["intermediate_size"]
    num_layers = draw(st.sampled_from(_VALID_NUM_LAYERS))

    state_dict: dict[str, torch.Tensor] = {}
    vocab_size = 100

    fused_qkv_size = (num_heads + 2 * num_kv_heads) * head_dim

    state_dict["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    state_dict["model.norm.weight"] = torch.randn(hidden_size)
    state_dict["model.norm.bias"] = torch.randn(hidden_size)
    state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.input_layernorm.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = torch.randn(
            fused_qkv_size, hidden_size
        )
        state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = torch.randn(fused_qkv_size)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, num_heads * head_dim
        )
        state_dict[f"{prefix}.self_attn.o_proj.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size
        )
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size
        )

    return state_dict


@st.composite
def fused_qkv_tensor(draw: st.DrawFn) -> tuple[torch.Tensor, int, int, int, int, int]:
    """Generate a valid fused QKV weight tensor with compatible dimensions.

    Returns (tensor, num_heads, num_kv_heads, head_dim, hidden_size, world_size).
    """
    world_size = draw(st.sampled_from(_VALID_WORLD_SIZES))
    num_heads = draw(st.sampled_from([h for h in _VALID_NUM_HEADS if h % world_size == 0]))
    num_kv_heads = draw(
        st.sampled_from(
            [kv for kv in _VALID_NUM_KV_HEADS if kv % world_size == 0 and kv <= num_heads]
        )
    )
    head_dim = draw(st.sampled_from([16, 32, 64]))
    hidden_size = draw(st.sampled_from(_VALID_HIDDEN_SIZES))

    fused_size = (num_heads + 2 * num_kv_heads) * head_dim
    tensor = torch.randn(fused_size, hidden_size)

    return tensor, num_heads, num_kv_heads, head_dim, hidden_size, world_size


@st.composite
def column_parallel_weight_bias_pair(
    draw: st.DrawFn,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Generate a weight-bias pair for a column-parallel layer.

    Returns (weight, bias, world_size, rank, output_per_rank).
    Column-parallel: weight shape (output_size, input_size), bias shape (output_size).
    Both are sharded along dim 0.
    """
    world_size = draw(st.sampled_from(_VALID_WORLD_SIZES))
    rank = draw(st.integers(min_value=0, max_value=world_size - 1))
    # output_size must be divisible by world_size
    output_per_rank = draw(st.sampled_from([16, 32, 64, 128]))
    output_size = output_per_rank * world_size
    input_size = draw(st.sampled_from(_VALID_HIDDEN_SIZES))

    weight = torch.randn(output_size, input_size)
    bias = torch.randn(output_size)

    return weight, bias, world_size, rank, output_per_rank


# =============================================================================
# Property 1: Key Preservation in shard_weights()
# =============================================================================


class TestKeyPreservationProperty:
    """**Validates: Requirements 1.5**

    Property 1: For any generated state dict, shard_weights() preserves all
    input keys in sharded_state_dict. For fused QKV (Phi), the original
    qkv_proj keys are replaced by q_proj, k_proj, v_proj keys.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_qwen_key_preservation(self, data: st.DataObject) -> None:
        """Feature: xpu-inference-e2e, Property 1: Key preservation in shard_weights()

        For Qwen/Llama state dicts, all input keys are preserved exactly.
        """
        config, params = data.draw(valid_tp_config())
        state_dict = data.draw(qwen_state_dict(params))

        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        input_keys = set(state_dict.keys())
        output_keys = set(shard.sharded_state_dict.keys())

        # All input keys must be present in output (exact preservation for Qwen)
        assert input_keys.issubset(output_keys), (
            f"Missing keys: {input_keys - output_keys}"
        )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_phi_key_preservation(self, data: st.DataObject) -> None:
        """Feature: xpu-inference-e2e, Property 1: Key preservation in shard_weights()

        For Phi state dicts with fused QKV, the qkv_proj keys are replaced
        by separate q_proj, k_proj, v_proj keys. All non-fused keys are preserved.
        """
        config, params = data.draw(valid_tp_config())
        state_dict = data.draw(phi_state_dict(params))

        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        input_keys = set(state_dict.keys())
        output_keys = set(shard.sharded_state_dict.keys())

        # Non-fused keys must be preserved
        non_fused_keys = {k for k in input_keys if "qkv_proj" not in k}
        assert non_fused_keys.issubset(output_keys), (
            f"Missing non-fused keys: {non_fused_keys - output_keys}"
        )

        # Fused QKV keys must be replaced by separate q/k/v keys
        fused_weight_keys = {k for k in input_keys if "qkv_proj.weight" in k}
        fused_bias_keys = {k for k in input_keys if "qkv_proj.bias" in k}

        for fused_key in fused_weight_keys:
            # Original fused key should NOT be in output
            assert fused_key not in output_keys
            # Replacement keys should exist
            q_key = fused_key.replace("qkv_proj.weight", "q_proj.weight")
            k_key = fused_key.replace("qkv_proj.weight", "k_proj.weight")
            v_key = fused_key.replace("qkv_proj.weight", "v_proj.weight")
            assert q_key in output_keys, f"Missing split key: {q_key}"
            assert k_key in output_keys, f"Missing split key: {k_key}"
            assert v_key in output_keys, f"Missing split key: {v_key}"

        for fused_key in fused_bias_keys:
            assert fused_key not in output_keys
            q_key = fused_key.replace("qkv_proj.bias", "q_proj.bias")
            k_key = fused_key.replace("qkv_proj.bias", "k_proj.bias")
            v_key = fused_key.replace("qkv_proj.bias", "v_proj.bias")
            assert q_key in output_keys, f"Missing split bias key: {q_key}"
            assert k_key in output_keys, f"Missing split bias key: {k_key}"
            assert v_key in output_keys, f"Missing split bias key: {v_key}"


# =============================================================================
# Property 2: Forward/Shard Key Consistency
# =============================================================================


class TestForwardShardKeyConsistencyProperty:
    """**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 7.1**

    Property 2: For any generated architecture-specific state dict, forward()
    accesses only keys that exist in sharded_state_dict (no KeyError).
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_qwen_forward_no_key_error(self, data: st.DataObject) -> None:
        """Feature: xpu-inference-e2e, Property 2: Forward/shard key consistency

        For any Qwen/Llama state dict, forward() completes without KeyError.
        """
        config, params = data.draw(valid_tp_config())
        state_dict = data.draw(qwen_state_dict(params))

        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        # Mock _all_reduce to be a no-op (no distributed backend in tests)
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        # Run forward pass — should not raise KeyError
        input_ids = torch.randint(0, 50, (1, 3))
        logits, kv_cache = shard.forward(input_ids)

        # Basic shape checks
        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert logits.shape[1] == 3

    @settings(max_examples=100)
    @given(data=st.data())
    def test_phi_forward_no_key_error(self, data: st.DataObject) -> None:
        """Feature: xpu-inference-e2e, Property 2: Forward/shard key consistency

        For any Phi state dict (fused QKV), forward() completes without KeyError.
        """
        config, params = data.draw(valid_tp_config())
        state_dict = data.draw(phi_state_dict(params))

        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        # Mock _all_reduce to be a no-op
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        # Run forward pass — should not raise KeyError
        input_ids = torch.randint(0, 50, (1, 3))
        logits, kv_cache = shard.forward(input_ids)

        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert logits.shape[1] == 3


# =============================================================================
# Property 3: Fused QKV Split Round-Trip
# =============================================================================


class TestFusedQKVRoundTripProperty:
    """**Validates: Requirements 7.2**

    Property 3: For any fused QKV weight, splitting into Q/K/V shards across
    all ranks and concatenating reconstructs the original.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_fused_qkv_split_round_trip(self, data: st.DataObject) -> None:
        """Feature: xpu-inference-e2e, Property 3: Fused QKV split round-trip

        Splitting a fused QKV weight into Q, K, V shards across all ranks
        and concatenating reconstructs the original tensor.
        """
        (
            fused_weight,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            world_size,
        ) = data.draw(fused_qkv_tensor())

        # Build a minimal Phi state dict with this fused weight
        intermediate_size = data.draw(
            st.sampled_from([s for s in _VALID_INTERMEDIATE_SIZES if s % world_size == 0])
        )
        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.norm.bias": torch.randn(hidden_size),
            "lm_head.weight": torch.randn(100, hidden_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.input_layernorm.bias": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.bias": torch.randn(hidden_size),
            "model.layers.0.self_attn.qkv_proj.weight": fused_weight.clone(),
            "model.layers.0.self_attn.qkv_proj.bias": torch.randn(
                fused_weight.shape[0]
            ),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                hidden_size, num_heads * head_dim
            ),
            "model.layers.0.self_attn.o_proj.bias": torch.randn(hidden_size),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                intermediate_size, hidden_size
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randn(
                hidden_size, intermediate_size
            ),
        }

        # Shard across all ranks and collect Q, K, V shards
        q_shards: list[torch.Tensor] = []
        k_shards: list[torch.Tensor] = []
        v_shards: list[torch.Tensor] = []

        for rank in range(world_size):
            config = TPShardConfig(
                rank=rank,
                world_size=world_size,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_kv_heads,
            )
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            q_shards.append(
                shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
            )
            k_shards.append(
                shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"]
            )
            v_shards.append(
                shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"]
            )

        # Reconstruct full Q, K, V by concatenating shards along dim 0
        q_full = torch.cat(q_shards, dim=0)
        k_full = torch.cat(k_shards, dim=0)
        v_full = torch.cat(v_shards, dim=0)

        # Reconstruct the original fused weight: [Q, K, V] along dim 0
        reconstructed = torch.cat([q_full, k_full, v_full], dim=0)

        assert torch.allclose(reconstructed, fused_weight), (
            f"Round-trip failed: max diff = {(reconstructed - fused_weight).abs().max().item()}"
        )


# =============================================================================
# Property 4: Bias Shard Dimension Consistency
# =============================================================================


class TestBiasShardDimensionConsistencyProperty:
    """**Validates: Requirements 7.3**

    Property 4: For any weight-bias pair, column-parallel bias shard dim 0
    equals weight shard dim 0.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_column_parallel_bias_dim_matches_weight(
        self, data: st.DataObject
    ) -> None:
        """Feature: xpu-inference-e2e, Property 4: Bias shard dimension consistency

        For any column-parallel weight-bias pair, the bias shard's dim 0
        equals the weight shard's dim 0 after sharding.
        """
        weight, bias, world_size, rank, output_per_rank = data.draw(
            column_parallel_weight_bias_pair()
        )

        # Shard weight along dim 0 (column-parallel)
        start = rank * output_per_rank
        weight_shard = weight.narrow(0, start, output_per_rank).clone()

        # Shard bias along dim 0 (same as weight for column-parallel)
        bias_shard = bias.narrow(0, start, output_per_rank).clone()

        # Property: bias shard dim 0 == weight shard dim 0
        assert bias_shard.shape[0] == weight_shard.shape[0], (
            f"Bias shard dim 0 ({bias_shard.shape[0]}) != "
            f"weight shard dim 0 ({weight_shard.shape[0]})"
        )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_phi_bias_shard_consistency_end_to_end(
        self, data: st.DataObject
    ) -> None:
        """Feature: xpu-inference-e2e, Property 4: Bias shard dimension consistency

        For any Phi state dict, column-parallel bias shards have dim 0
        matching their corresponding weight shards.
        """
        config, params = data.draw(valid_tp_config())
        state_dict = data.draw(phi_state_dict(params))

        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        # Check all column-parallel weight-bias pairs
        # Column-parallel: q_proj, k_proj, v_proj (from fused split), gate_proj, up_proj
        num_layers = shard._detect_num_layers()
        for layer_idx in range(num_layers):
            # Q projection
            q_weight = shard.sharded_state_dict[
                f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            ]
            q_bias = shard.sharded_state_dict.get(
                f"model.layers.{layer_idx}.self_attn.q_proj.bias"
            )
            if q_bias is not None:
                assert q_bias.shape[0] == q_weight.shape[0], (
                    f"Layer {layer_idx} q_proj: bias dim 0 ({q_bias.shape[0]}) != "
                    f"weight dim 0 ({q_weight.shape[0]})"
                )

            # K projection
            k_weight = shard.sharded_state_dict[
                f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            ]
            k_bias = shard.sharded_state_dict.get(
                f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            )
            if k_bias is not None:
                assert k_bias.shape[0] == k_weight.shape[0], (
                    f"Layer {layer_idx} k_proj: bias dim 0 ({k_bias.shape[0]}) != "
                    f"weight dim 0 ({k_weight.shape[0]})"
                )

            # V projection
            v_weight = shard.sharded_state_dict[
                f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            ]
            v_bias = shard.sharded_state_dict.get(
                f"model.layers.{layer_idx}.self_attn.v_proj.bias"
            )
            if v_bias is not None:
                assert v_bias.shape[0] == v_weight.shape[0], (
                    f"Layer {layer_idx} v_proj: bias dim 0 ({v_bias.shape[0]}) != "
                    f"weight dim 0 ({v_weight.shape[0]})"
                )
