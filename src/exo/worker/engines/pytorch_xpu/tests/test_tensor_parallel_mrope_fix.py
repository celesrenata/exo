"""Property-based tests for the tensor-parallel MRoPE fix.

Tests both the bug condition (task 1) and preservation properties (task 2)
for the streaming loader MRoPE rotary embedding fix.

Spec: tensor-parallel-mrope-fix
"""

from __future__ import annotations

import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
)


# =============================================================================
# Task 1: Bug Condition Exploration Tests
# =============================================================================


class TestBugConditionExploration:
    """Property 1: Bug Condition — Streaming MRoPE Models Get Native Rotary Embedding.

    These tests encode the EXPECTED behavior after the fix:
    - _native_rotary_emb is NOT None when native_rotary_emb is passed
    - q_proj.bias shard size matches weight shard size for doubled Q projections
    - RuntimeError IS raised when MRoPE config has _native_rotary_emb = None

    On UNFIXED code, these tests FAIL (proving the bug exists).
    On FIXED code, these tests PASS (confirming the fix works).

    **Validates: Requirements 2.1, 2.2, 2.4, 2.5**
    """

    def test_native_rotary_emb_stored_when_provided(self) -> None:
        """Streaming MRoPE models get native rotary embedding when provided.

        When native_rotary_emb is passed to TensorParallelShard, it is stored
        in self._native_rotary_emb and is not None.

        **Validates: Requirements 2.1, 2.2**
        """
        # MRoPE config (Qwen3.5-27B style)
        config = TPShardConfig(
            rank=0,
            world_size=4,
            hidden_size=3584,
            num_attention_heads=28,
            head_dim=256,
            intermediate_size=18944,
            num_key_value_heads=4,
            rope_scaling={"type": "mrope", "mrope_section": [11, 11, 10]},
            mrope_interleaved=True,
        )

        # Create a mock rotary embedding (any module with a .to() method)
        class MockRotaryEmbedding(torch.nn.Module):
            pass

        mock_rotary = MockRotaryEmbedding()

        # Streaming path: model is a dict, pre_sharded=True, native_rotary_emb provided
        shard = TensorParallelShard(
            model={},
            config=config,
            device="cpu",
            pre_sharded=True,
            native_rotary_emb=mock_rotary,
        )

        # Expected behavior: _native_rotary_emb is NOT None
        assert shard._native_rotary_emb is not None, (
            "Bug condition: _native_rotary_emb is None for streaming MRoPE model "
            "even when native_rotary_emb was provided"
        )
        assert "RotaryEmbedding" in type(shard._native_rotary_emb).__name__ or isinstance(
            shard._native_rotary_emb, torch.nn.Module
        ), (
            f"Expected a rotary embedding module, got {type(shard._native_rotary_emb).__name__}"
        )

    @given(
        world_size=st.integers(min_value=2, max_value=8),
        head_multiplier=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=50, deadline=None)
    def test_q_proj_bias_shard_matches_weight_shard_for_doubled_q(
        self, world_size: int, head_multiplier: int
    ) -> None:
        """q_proj.bias shard size matches q_proj.weight shard size for doubled Q.

        For doubled Q projections (Q + gate), the bias tensor has size
        num_heads * head_dim * 2. The shard size must be heads_per_rank * head_dim * 2,
        matching the weight sharding logic.

        **Validates: Requirements 2.5**
        """
        num_heads = head_multiplier * world_size
        head_dim = 128
        hidden_size = num_heads * head_dim
        num_kv_heads = world_size  # Minimum valid: 1 per rank
        intermediate_size = world_size * 256

        config = TPShardConfig(
            rank=0,
            world_size=world_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_key_value_heads=num_kv_heads,
            mrope_interleaved=False,
        )

        # Doubled Q projection: weight and bias have doubled output dim
        doubled_out_dim = num_heads * head_dim * 2
        q_weight = torch.randn(doubled_out_dim, hidden_size)
        q_bias = torch.randn(doubled_out_dim)

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": q_weight,
            "model.layers.0.self_attn.q_proj.bias": q_bias,
            "model.layers.0.self_attn.k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Get sharded weight and bias
        q_weight_shard = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        q_bias_shard = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.bias"]

        # Expected: bias shard size matches weight shard size (both doubled)
        heads_per_rank = num_heads // world_size
        expected_shard_size = heads_per_rank * head_dim * 2

        assert q_weight_shard.shape[0] == expected_shard_size, (
            f"q_proj.weight shard dim {q_weight_shard.shape[0]} != expected {expected_shard_size}"
        )
        assert q_bias_shard.shape[0] == expected_shard_size, (
            f"Bug condition: q_proj.bias shard dim {q_bias_shard.shape[0]} != "
            f"q_proj.weight shard dim {expected_shard_size}. "
            f"Bias shard must match weight shard for doubled Q projections."
        )

    def test_runtime_error_raised_for_mrope_without_native_rotary_emb(self) -> None:
        """RuntimeError is raised when MRoPE config has _native_rotary_emb = None.

        When an MRoPE model (mrope_interleaved=True) is constructed without
        native_rotary_emb, the system must raise RuntimeError instead of
        silently falling back to incorrect 1D RoPE.

        **Validates: Requirements 2.4**
        """
        # MRoPE config without native rotary embedding
        config = TPShardConfig(
            rank=0,
            world_size=4,
            hidden_size=3584,
            num_attention_heads=28,
            head_dim=256,
            intermediate_size=18944,
            num_key_value_heads=4,
            rope_scaling={"type": "mrope", "mrope_section": [11, 11, 10]},
            mrope_interleaved=True,
        )

        # Streaming path: model is a dict, pre_sharded=True, NO native_rotary_emb
        with pytest.raises(RuntimeError, match="MRoPE model detected"):
            TensorParallelShard(
                model={},
                config=config,
                device="cpu",
                pre_sharded=True,
                # native_rotary_emb NOT provided — should raise RuntimeError
            )


# =============================================================================
# Strategies
# =============================================================================


@st.composite
def non_mrope_tp_shard_configs(draw: st.DrawFn) -> TPShardConfig:
    """Generate valid TPShardConfig instances WITHOUT MRoPE.

    Constraints:
    - num_attention_heads divisible by world_size
    - num_key_value_heads divisible by world_size
    - intermediate_size divisible by world_size
    - mrope_interleaved=False
    - rope_scaling is None or rope_scaling.type != "mrope"
    """
    world_size = draw(st.integers(min_value=2, max_value=8))

    # num_heads must be divisible by world_size
    head_multiplier = draw(st.integers(min_value=1, max_value=16))
    num_heads = head_multiplier * world_size

    # num_kv_heads must divide num_heads and be divisible by world_size
    valid_kv_heads = [
        k for k in range(world_size, num_heads + 1, world_size)
        if num_heads % k == 0
    ]
    assume(len(valid_kv_heads) > 0)
    num_kv_heads = draw(st.sampled_from(valid_kv_heads))

    head_dim = draw(st.sampled_from([32, 64, 128, 256]))
    hidden_size = num_heads * head_dim

    # intermediate_size must be divisible by world_size
    intermediate_multiplier = draw(st.integers(min_value=1, max_value=16))
    intermediate_size = intermediate_multiplier * world_size * 64

    rank = draw(st.integers(min_value=0, max_value=world_size - 1))

    # Non-MRoPE rope_scaling: None or a non-mrope type
    rope_scaling_choice = draw(st.sampled_from([None, {"type": "linear", "factor": 2.0}, {"type": "yarn", "factor": 4.0}]))

    return TPShardConfig(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_kv_heads,
        rope_scaling=rope_scaling_choice,
        mrope_interleaved=False,
    )


@st.composite
def shardable_tensor_configs(draw: st.DrawFn) -> dict:
    """Generate random tensor shapes for shardable parameters.

    Returns a dict with config and parameter info for testing deterministic sharding.
    """
    world_size = draw(st.integers(min_value=2, max_value=8))

    head_multiplier = draw(st.integers(min_value=1, max_value=8))
    num_heads = head_multiplier * world_size

    valid_kv_heads = [
        k for k in range(world_size, num_heads + 1, world_size)
        if num_heads % k == 0
    ]
    assume(len(valid_kv_heads) > 0)
    num_kv_heads = draw(st.sampled_from(valid_kv_heads))

    head_dim = draw(st.sampled_from([32, 64, 128]))
    hidden_size = num_heads * head_dim

    intermediate_multiplier = draw(st.integers(min_value=1, max_value=8))
    intermediate_size = intermediate_multiplier * world_size * 64

    rank = draw(st.integers(min_value=0, max_value=world_size - 1))

    # Choose which parameter type to test
    param_type = draw(st.sampled_from([
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "o_proj.weight", "gate_proj.weight", "up_proj.weight",
        "down_proj.weight",
    ]))

    return {
        "world_size": world_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rank": rank,
        "param_type": param_type,
    }


# =============================================================================
# Task 2: Preservation Property Tests
# =============================================================================


class TestPreservationNonMRoPESharding:
    """Property 2: Preservation — Non-MRoPE Sharding and HF Model Path Unchanged.

    For all TPShardConfig instances WITHOUT MRoPE (mrope_interleaved=False, no mrope
    rope_scaling), `_shard_parameter` produces shards with correct output dimensions
    matching `heads_per_rank * head_dim` for Q, `kv_heads_per_rank * head_dim` for K/V,
    and `intermediate_per_rank` for MLP.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
    """

    @given(config=non_mrope_tp_shard_configs())
    @settings(max_examples=100, deadline=None)
    def test_non_mrope_q_proj_shard_dimensions(self, config: TPShardConfig) -> None:
        """Q projection shard has correct dimensions for non-MRoPE configs.

        For standard (non-doubled) Q projections, shard output dim = heads_per_rank * head_dim.

        **Validates: Requirements 3.4**
        """
        heads_per_rank = config.heads_per_rank
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        # Create a standard q_proj weight: (num_heads * head_dim, hidden_size)
        q_weight = torch.randn(num_heads * head_dim, hidden_size)

        # Create a minimal state dict and shard
        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": q_weight,
            "model.layers.0.self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, config.intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        q_shard = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        expected_out_dim = heads_per_rank * head_dim
        assert q_shard.shape[0] == expected_out_dim, (
            f"Q shard output dim {q_shard.shape[0]} != expected {expected_out_dim}"
        )
        assert q_shard.shape[1] == hidden_size, (
            f"Q shard input dim {q_shard.shape[1]} != hidden_size {hidden_size}"
        )

    @given(config=non_mrope_tp_shard_configs())
    @settings(max_examples=100, deadline=None)
    def test_non_mrope_kv_proj_shard_dimensions(self, config: TPShardConfig) -> None:
        """K and V projection shards have correct dimensions for non-MRoPE configs.

        Shard output dim = kv_heads_per_rank * head_dim.

        **Validates: Requirements 3.5**
        """
        kv_heads_per_rank = config.kv_heads_per_rank
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, config.intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        expected_kv_dim = kv_heads_per_rank * head_dim

        k_shard = shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"]
        assert k_shard.shape[0] == expected_kv_dim, (
            f"K shard output dim {k_shard.shape[0]} != expected {expected_kv_dim}"
        )
        assert k_shard.shape[1] == hidden_size

        v_shard = shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"]
        assert v_shard.shape[0] == expected_kv_dim, (
            f"V shard output dim {v_shard.shape[0]} != expected {expected_kv_dim}"
        )
        assert v_shard.shape[1] == hidden_size

    @given(config=non_mrope_tp_shard_configs())
    @settings(max_examples=100, deadline=None)
    def test_non_mrope_o_proj_shard_dimensions(self, config: TPShardConfig) -> None:
        """O projection shard has correct dimensions for non-MRoPE configs.

        o_proj is row-parallel: shard input dim = heads_per_rank * head_dim.

        **Validates: Requirements 3.5**
        """
        heads_per_rank = config.heads_per_rank
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, config.intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        o_shard = shard.sharded_state_dict["model.layers.0.self_attn.o_proj.weight"]
        expected_input_dim = heads_per_rank * head_dim
        assert o_shard.shape[0] == hidden_size, (
            f"O shard output dim {o_shard.shape[0]} != hidden_size {hidden_size}"
        )
        assert o_shard.shape[1] == expected_input_dim, (
            f"O shard input dim {o_shard.shape[1]} != expected {expected_input_dim}"
        )

    @given(config=non_mrope_tp_shard_configs())
    @settings(max_examples=100, deadline=None)
    def test_non_mrope_mlp_shard_dimensions(self, config: TPShardConfig) -> None:
        """MLP projection shards have correct dimensions for non-MRoPE configs.

        gate_proj, up_proj: shard output dim = intermediate_per_rank
        down_proj: shard input dim = intermediate_per_rank

        **Validates: Requirements 3.6**
        """
        intermediate_per_rank = config.intermediate_per_rank
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, config.intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        gate_shard = shard.sharded_state_dict["model.layers.0.mlp.gate_proj.weight"]
        assert gate_shard.shape[0] == intermediate_per_rank, (
            f"gate_proj shard dim {gate_shard.shape[0]} != expected {intermediate_per_rank}"
        )
        assert gate_shard.shape[1] == hidden_size

        up_shard = shard.sharded_state_dict["model.layers.0.mlp.up_proj.weight"]
        assert up_shard.shape[0] == intermediate_per_rank, (
            f"up_proj shard dim {up_shard.shape[0]} != expected {intermediate_per_rank}"
        )
        assert up_shard.shape[1] == hidden_size

        down_shard = shard.sharded_state_dict["model.layers.0.mlp.down_proj.weight"]
        assert down_shard.shape[0] == hidden_size
        assert down_shard.shape[1] == intermediate_per_rank, (
            f"down_proj shard dim {down_shard.shape[1]} != expected {intermediate_per_rank}"
        )

    @given(config=non_mrope_tp_shard_configs())
    @settings(max_examples=100, deadline=None)
    def test_non_mrope_doubled_q_proj_weight_shard(self, config: TPShardConfig) -> None:
        """Doubled Q projection weight sharding uses shape-based detection correctly.

        For doubled Q projections (Q + gate), the weight has shape
        (num_heads * head_dim * 2, hidden_size) and the shard size is
        heads_per_rank * head_dim * 2.

        **Validates: Requirements 3.4**
        """
        heads_per_rank = config.heads_per_rank
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        # Doubled Q projection: output dim is num_heads * head_dim * 2
        doubled_out_dim = num_heads * head_dim * 2
        q_weight_doubled = torch.randn(doubled_out_dim, hidden_size)

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": q_weight_doubled,
            "model.layers.0.self_attn.k_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(config.num_key_value_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(config.intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, config.intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        q_shard = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        expected_doubled_shard = heads_per_rank * head_dim * 2
        assert q_shard.shape[0] == expected_doubled_shard, (
            f"Doubled Q shard output dim {q_shard.shape[0]} != expected {expected_doubled_shard}"
        )
        assert q_shard.shape[1] == hidden_size


class TestPreservationShardingDeterminism:
    """Property 2: Preservation — Sharding is deterministic and dimensions are correct.

    For all random tensor shapes for shardable parameters, sharding is deterministic
    and shard dimensions equal `full_dim / world_size`.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
    """

    @given(data=shardable_tensor_configs())
    @settings(max_examples=100, deadline=None)
    def test_sharding_is_deterministic(self, data: dict) -> None:
        """Sharding the same tensor twice produces identical results.

        **Validates: Requirements 3.1, 3.2**
        """
        config = TPShardConfig(
            rank=data["rank"],
            world_size=data["world_size"],
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            num_key_value_heads=data["num_kv_heads"],
            mrope_interleaved=False,
        )

        hidden_size = data["hidden_size"]
        num_heads = data["num_heads"]
        num_kv_heads = data["num_kv_heads"]
        head_dim = data["head_dim"]
        intermediate_size = data["intermediate_size"]

        # Build a state dict with all parameter types
        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        # Shard twice with the same config
        shard1 = TensorParallelShard(state_dict, config, device="cpu")
        shard2 = TensorParallelShard(state_dict, config, device="cpu")

        param_key = f"model.layers.0.{'self_attn' if 'proj' in data['param_type'] and 'gate' not in data['param_type'] and 'up' not in data['param_type'] and 'down' not in data['param_type'] else 'mlp'}.{data['param_type']}"

        # Determine the correct key based on param_type
        if data["param_type"] in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
            key = f"model.layers.0.self_attn.{data['param_type']}"
        else:
            key = f"model.layers.0.mlp.{data['param_type']}"

        t1 = shard1.sharded_state_dict[key]
        t2 = shard2.sharded_state_dict[key]

        assert torch.equal(t1, t2), (
            f"Sharding is not deterministic for {key}: "
            f"max diff = {(t1 - t2).abs().max().item()}"
        )

    @given(data=shardable_tensor_configs())
    @settings(max_examples=100, deadline=None)
    def test_shard_dimensions_equal_full_div_world_size(self, data: dict) -> None:
        """Shard dimensions equal full_dim / world_size for all shardable parameters.

        **Validates: Requirements 3.3, 3.4, 3.5, 3.6**
        """
        world_size = data["world_size"]
        config = TPShardConfig(
            rank=data["rank"],
            world_size=world_size,
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            num_key_value_heads=data["num_kv_heads"],
            mrope_interleaved=False,
        )

        hidden_size = data["hidden_size"]
        num_heads = data["num_heads"]
        num_kv_heads = data["num_kv_heads"]
        head_dim = data["head_dim"]
        intermediate_size = data["intermediate_size"]

        state_dict: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, hidden_size),
            "model.norm.weight": torch.randn(hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size),
            "model.layers.0.input_layernorm.weight": torch.randn(hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden_size),
        }

        shard = TensorParallelShard(state_dict, config, device="cpu")

        param_type = data["param_type"]
        if param_type in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
            key = f"model.layers.0.self_attn.{param_type}"
        else:
            key = f"model.layers.0.mlp.{param_type}"

        sharded_tensor = shard.sharded_state_dict[key]

        # Determine expected shard dimension based on parameter type
        if param_type == "q_proj.weight":
            # Column-parallel: output dim sharded
            full_dim = num_heads * head_dim
            expected_shard_dim = full_dim // world_size
            assert sharded_tensor.shape[0] == expected_shard_dim, (
                f"q_proj shard dim[0] {sharded_tensor.shape[0]} != {expected_shard_dim}"
            )
        elif param_type in ("k_proj.weight", "v_proj.weight"):
            # Column-parallel: output dim sharded
            full_dim = num_kv_heads * head_dim
            expected_shard_dim = full_dim // world_size
            assert sharded_tensor.shape[0] == expected_shard_dim, (
                f"{param_type} shard dim[0] {sharded_tensor.shape[0]} != {expected_shard_dim}"
            )
        elif param_type == "o_proj.weight":
            # Row-parallel: input dim sharded (dim 1)
            full_dim = num_heads * head_dim
            expected_shard_dim = full_dim // world_size
            assert sharded_tensor.shape[1] == expected_shard_dim, (
                f"o_proj shard dim[1] {sharded_tensor.shape[1]} != {expected_shard_dim}"
            )
        elif param_type in ("gate_proj.weight", "up_proj.weight"):
            # Column-parallel: output dim sharded
            full_dim = intermediate_size
            expected_shard_dim = full_dim // world_size
            assert sharded_tensor.shape[0] == expected_shard_dim, (
                f"{param_type} shard dim[0] {sharded_tensor.shape[0]} != {expected_shard_dim}"
            )
        elif param_type == "down_proj.weight":
            # Row-parallel: input dim sharded (dim 1)
            full_dim = intermediate_size
            expected_shard_dim = full_dim // world_size
            assert sharded_tensor.shape[1] == expected_shard_dim, (
                f"down_proj shard dim[1] {sharded_tensor.shape[1]} != {expected_shard_dim}"
            )
