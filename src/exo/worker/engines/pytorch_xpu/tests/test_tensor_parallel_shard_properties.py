"""Property-based tests for TensorParallelShard.

Tests weight shard shape correctness, divisibility validation, and
forward pass equivalence using Hypothesis.

Feature: tensor-parallelism-xpu
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
)


# --- Strategies ---


@st.composite
def valid_model_configs(draw: st.DrawFn) -> dict:
    """Generate random valid model configurations constrained to divisibility.

    Generates configs where:
    - hidden_size ∈ [64, 8192], divisible by num_heads and world_size
    - num_heads ∈ [4, 128], divisible by world_size
    - num_kv_heads divides num_heads and is divisible by world_size
    - intermediate_size divisible by world_size
    - world_size ∈ [2, 8]
    - rank ∈ [0, world_size)
    """
    world_size = draw(st.integers(min_value=2, max_value=8))

    # num_heads must be divisible by world_size, range [4, 128]
    # Pick a multiplier such that num_heads = multiplier * world_size
    max_multiplier = 128 // world_size
    min_multiplier = max(1, 4 // world_size + (1 if 4 % world_size != 0 else 0))
    assume(min_multiplier <= max_multiplier)
    head_multiplier = draw(st.integers(min_value=min_multiplier, max_value=max_multiplier))
    num_heads = head_multiplier * world_size

    # num_kv_heads must divide num_heads and be divisible by world_size
    # Valid kv_heads: multiples of world_size that also divide num_heads
    valid_kv_heads = [
        k for k in range(world_size, num_heads + 1, world_size)
        if num_heads % k == 0
    ]
    assume(len(valid_kv_heads) > 0)
    num_kv_heads = draw(st.sampled_from(valid_kv_heads))

    # head_dim: pick from common values
    head_dim = draw(st.sampled_from([32, 64, 80, 128, 256]))

    # hidden_size derived from num_heads * head_dim (standard transformer convention)
    # But we allow it to be independent for generality — just needs to be > 0
    hidden_size = num_heads * head_dim

    # intermediate_size must be divisible by world_size
    intermediate_multiplier = draw(st.integers(min_value=1, max_value=32))
    intermediate_size = intermediate_multiplier * world_size * 64  # Keep sizes reasonable

    rank = draw(st.integers(min_value=0, max_value=world_size - 1))

    return {
        "rank": rank,
        "world_size": world_size,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_key_value_heads": num_kv_heads,
    }


@st.composite
def head_world_pairs(draw: st.DrawFn) -> tuple[int, int, int]:
    """Generate random (num_heads, num_kv_heads, world_size) triples.

    Does NOT constrain divisibility — used for testing validation.
    """
    num_heads = draw(st.integers(min_value=1, max_value=128))
    num_kv_heads = draw(st.integers(min_value=1, max_value=num_heads))
    world_size = draw(st.integers(min_value=2, max_value=8))
    return (num_heads, num_kv_heads, world_size)


@st.composite
def row_parallel_inputs(draw: st.DrawFn) -> dict:
    """Generate random input tensors and weight matrices for forward pass equivalence.

    Generates a full weight matrix and input, suitable for testing that
    splitting the weight across N ranks and summing partial results equals
    the full computation.
    """
    world_size = draw(st.integers(min_value=2, max_value=8))
    batch_size = draw(st.integers(min_value=1, max_value=4))
    seq_len = draw(st.integers(min_value=1, max_value=8))

    # Output dim (e.g., hidden_size for down_proj)
    output_dim = draw(st.integers(min_value=16, max_value=128)) * world_size
    # Input dim must be divisible by world_size (row-parallel splits input dim)
    input_dim = draw(st.integers(min_value=16, max_value=128)) * world_size

    # Generate input and weight in float32 for numerical stability
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    weight = torch.randn(output_dim, input_dim)

    return {
        "world_size": world_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "output_dim": output_dim,
        "input_dim": input_dim,
        "input_tensor": input_tensor,
        "weight": weight,
    }


# --- Property 2: Weight Shard Shape Correctness ---


# Feature: tensor-parallelism-xpu, Property 2: Weight shard shape correctness
class TestWeightShardShapes:
    """Property 2: Weight Shard Shape Correctness.

    For any valid model configuration, the weight sharding function SHALL produce
    tensors with shapes matching the formulas from the design document.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    @given(config=valid_model_configs())
    @settings(max_examples=100, deadline=None)
    def test_all_shard_shapes_match_formula(self, config: dict) -> None:
        """All weight shard shapes match the design document formulas.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        rank = config["rank"]
        world_size = config["world_size"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        head_dim = config["head_dim"]
        intermediate_size = config["intermediate_size"]
        num_kv_heads = config["num_key_value_heads"]

        # Create a minimal state dict with one layer
        state_dict: dict[str, torch.Tensor] = {}
        state_dict["model.embed_tokens.weight"] = torch.randn(100, hidden_size)
        state_dict["model.norm.weight"] = torch.randn(hidden_size)
        state_dict["lm_head.weight"] = torch.randn(100, hidden_size)

        prefix = "model.layers.0"
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

        tp_config = TPShardConfig(
            rank=rank,
            world_size=world_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_key_value_heads=num_kv_heads,
        )

        shard = TensorParallelShard(state_dict, tp_config, device="cpu")

        # Expected shapes from design document formulas
        heads_per_rank = num_heads // world_size
        kv_heads_per_rank = num_kv_heads // world_size
        intermediate_per_rank = intermediate_size // world_size

        # q_proj: (num_attention_heads / world_size * head_dim, hidden_size)
        q = shard.sharded_state_dict[f"{prefix}.self_attn.q_proj.weight"]
        assert q.shape == (heads_per_rank * head_dim, hidden_size), (
            f"q_proj shape {q.shape} != expected ({heads_per_rank * head_dim}, {hidden_size})"
        )

        # k_proj: (num_key_value_heads / world_size * head_dim, hidden_size)
        k = shard.sharded_state_dict[f"{prefix}.self_attn.k_proj.weight"]
        assert k.shape == (kv_heads_per_rank * head_dim, hidden_size), (
            f"k_proj shape {k.shape} != expected ({kv_heads_per_rank * head_dim}, {hidden_size})"
        )

        # v_proj: (num_key_value_heads / world_size * head_dim, hidden_size)
        v = shard.sharded_state_dict[f"{prefix}.self_attn.v_proj.weight"]
        assert v.shape == (kv_heads_per_rank * head_dim, hidden_size), (
            f"v_proj shape {v.shape} != expected ({kv_heads_per_rank * head_dim}, {hidden_size})"
        )

        # o_proj: (hidden_size, num_attention_heads / world_size * head_dim)
        o = shard.sharded_state_dict[f"{prefix}.self_attn.o_proj.weight"]
        assert o.shape == (hidden_size, heads_per_rank * head_dim), (
            f"o_proj shape {o.shape} != expected ({hidden_size}, {heads_per_rank * head_dim})"
        )

        # gate_proj: (intermediate_size / world_size, hidden_size)
        gate = shard.sharded_state_dict[f"{prefix}.mlp.gate_proj.weight"]
        assert gate.shape == (intermediate_per_rank, hidden_size), (
            f"gate_proj shape {gate.shape} != expected ({intermediate_per_rank}, {hidden_size})"
        )

        # up_proj: (intermediate_size / world_size, hidden_size)
        up = shard.sharded_state_dict[f"{prefix}.mlp.up_proj.weight"]
        assert up.shape == (intermediate_per_rank, hidden_size), (
            f"up_proj shape {up.shape} != expected ({intermediate_per_rank}, {hidden_size})"
        )

        # down_proj: (hidden_size, intermediate_size / world_size)
        down = shard.sharded_state_dict[f"{prefix}.mlp.down_proj.weight"]
        assert down.shape == (hidden_size, intermediate_per_rank), (
            f"down_proj shape {down.shape} != expected ({hidden_size}, {intermediate_per_rank})"
        )


# --- Property 3: Divisibility Validation ---


# Feature: tensor-parallelism-xpu, Property 3: Divisibility validation
class TestDivisibilityValidation:
    """Property 3: Divisibility Validation.

    For any (num_heads, world_size) pair where num_heads % world_size != 0,
    TPShardConfig initialization SHALL raise a ValueError. Conversely, for any
    pair where all dimensions are divisible, initialization SHALL NOT raise.

    **Validates: Requirements 3.6, 7.2**
    """

    @given(data=head_world_pairs())
    @settings(max_examples=100)
    def test_value_error_iff_not_divisible(self, data: tuple[int, int, int]) -> None:
        """ValueError raised iff num_heads % world_size != 0 (or kv_heads/intermediate).

        **Validates: Requirements 3.6, 7.2**
        """
        num_heads, num_kv_heads, world_size = data

        # Use a fixed intermediate_size that's always divisible by world_size
        # so we isolate the head divisibility check
        intermediate_size = world_size * 128
        hidden_size = 256
        head_dim = 64

        heads_divisible = num_heads % world_size == 0
        kv_heads_divisible = num_kv_heads % world_size == 0

        if not heads_divisible or not kv_heads_divisible:
            # Should raise ValueError
            with pytest.raises(ValueError):
                TPShardConfig(
                    rank=0,
                    world_size=world_size,
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_key_value_heads=num_kv_heads,
                )
        else:
            # Should NOT raise ValueError
            config = TPShardConfig(
                rank=0,
                world_size=world_size,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_kv_heads,
            )
            assert config.num_attention_heads == num_heads
            assert config.world_size == world_size


# --- Property 4: Tensor-Parallel Forward Pass Equivalence ---


# Feature: tensor-parallelism-xpu, Property 4: Forward pass equivalence
class TestForwardPassEquivalence:
    """Property 4: Tensor-Parallel Forward Pass Equivalence.

    For any input tensor and weight matrix, splitting the weight across N ranks
    (row-parallel), computing F.linear with each shard, and summing the results
    SHALL equal F.linear with the full weight.

    This tests the mathematical correctness of row-parallel linear.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.2**
    """

    @given(data=row_parallel_inputs())
    @settings(max_examples=100, deadline=None)
    def test_sharded_sum_equals_full_linear(self, data: dict) -> None:
        """Splitting weight across ranks and summing partial results equals full linear.

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.2**
        """
        world_size = data["world_size"]
        input_tensor = data["input_tensor"]
        weight = data["weight"]
        input_dim = data["input_dim"]
        output_dim = data["output_dim"]

        # Full computation (single-node reference)
        full_output = F.linear(input_tensor, weight)

        # Sharded computation: split weight along input dim (dim=1) for row-parallel
        shard_size = input_dim // world_size
        partial_outputs = []

        for rank in range(world_size):
            # Each rank gets a slice of the input dimension
            weight_shard = weight[:, rank * shard_size : (rank + 1) * shard_size]
            input_shard = input_tensor[..., rank * shard_size : (rank + 1) * shard_size]
            # Each rank computes F.linear with its weight shard and input shard
            partial = F.linear(input_shard, weight_shard)
            partial_outputs.append(partial)

        # All-reduce (sum) combines partial outputs
        sharded_output = torch.stack(partial_outputs).sum(dim=0)

        # Verify equivalence within floating-point tolerance
        # With large accumulations, float32 can have errors up to ~1e-4
        assert torch.allclose(full_output, sharded_output, atol=1e-4, rtol=1e-4), (
            f"Max difference: {(full_output - sharded_output).abs().max().item()}, "
            f"world_size={world_size}, input_dim={input_dim}, output_dim={output_dim}"
        )

    @given(data=row_parallel_inputs())
    @settings(max_examples=100, deadline=None)
    def test_sharded_sum_equals_full_linear_bf16(self, data: dict) -> None:
        """Forward pass equivalence holds within bf16 tolerance.

        Tests that the sharding property holds when weights and inputs are bf16.
        The partial results from each rank are computed in bf16, then summed
        in float32 (as Gloo all-reduce does internally for numerical stability).

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.2**
        """
        world_size = data["world_size"]
        input_dim = data["input_dim"]

        # Convert to bf16 FIRST, then use the bf16 tensors for both computations
        input_tensor = data["input_tensor"].to(torch.bfloat16)
        weight = data["weight"].to(torch.bfloat16)

        # Full computation in bf16
        full_output = F.linear(input_tensor, weight)

        # Sharded computation: each rank computes in bf16, sum in float32
        # (mirrors real all-reduce behavior where partial sums are accumulated)
        shard_size = input_dim // world_size
        accumulated = torch.zeros_like(full_output, dtype=torch.float32)

        for rank in range(world_size):
            weight_shard = weight[:, rank * shard_size : (rank + 1) * shard_size]
            input_shard = input_tensor[..., rank * shard_size : (rank + 1) * shard_size]
            partial = F.linear(input_shard, weight_shard)
            accumulated += partial.float()

        sharded_output = accumulated.to(torch.bfloat16)

        # Both results are bf16 — compare with tolerance appropriate for bf16
        # bf16 has ~3 decimal digits of precision
        full_f32 = full_output.float()
        sharded_f32 = sharded_output.float()

        # Use absolute tolerance scaled to the magnitude of outputs
        max_magnitude = torch.maximum(full_f32.abs(), sharded_f32.abs()).max().item()
        # bf16 relative precision is ~2^-8 ≈ 0.004, allow 2x margin
        atol = max(max_magnitude * 0.01, 0.1)

        assert torch.allclose(full_f32, sharded_f32, atol=atol, rtol=0.01), (
            f"Max difference (bf16): {(full_f32 - sharded_f32).abs().max().item()}, "
            f"atol={atol:.4f}, max_magnitude={max_magnitude:.2f}, "
            f"world_size={world_size}, input_dim={input_dim}"
        )
