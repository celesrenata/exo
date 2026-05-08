"""Unit tests for TensorParallelShard.__init__() and shard_weights().

Tests weight sharding correctness for Qwen3.5-4B-like model configurations
with tensor parallelism across 4 ranks.
"""

from __future__ import annotations

import torch
import pytest
import torch.nn.functional as F

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
    TPShardConfig,
    TensorParallelShard,
)


def _make_qwen_config(rank: int = 0, world_size: int = 4) -> TPShardConfig:
    """Create a Qwen3.5-4B-like TPShardConfig."""
    return TPShardConfig(
        rank=rank,
        world_size=world_size,
        hidden_size=2560,
        num_attention_heads=32,
        head_dim=80,  # 2560 / 32 = 80
        intermediate_size=6912,
        num_key_value_heads=8,
    )


def _make_fake_state_dict(
    hidden_size: int = 2560,
    num_heads: int = 32,
    head_dim: int = 80,
    num_kv_heads: int = 8,
    intermediate_size: int = 6912,
    num_layers: int = 2,
) -> dict[str, torch.Tensor]:
    """Create a fake state dict mimicking Qwen3.5-4B parameter names and shapes."""
    state_dict: dict[str, torch.Tensor] = {}

    # Embedding and final layers (redundant)
    state_dict["model.embed_tokens.weight"] = torch.randn(151936, hidden_size)
    state_dict["model.norm.weight"] = torch.randn(hidden_size)
    state_dict["lm_head.weight"] = torch.randn(151936, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Layer norms (redundant)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)

        # Attention projections
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

        # MLP projections
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


class TestTensorParallelShardInit:
    """Tests for TensorParallelShard.__init__()."""

    def test_accepts_state_dict(self) -> None:
        """TensorParallelShard accepts a plain dict as model input."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert len(shard.sharded_state_dict) == len(state_dict)

    def test_accepts_module_with_state_dict(self) -> None:
        """TensorParallelShard accepts an object with state_dict() method."""

        class FakeModule:
            def state_dict(self) -> dict[str, torch.Tensor]:
                return _make_fake_state_dict(num_layers=1)

        config = _make_qwen_config(rank=0)
        shard = TensorParallelShard(FakeModule(), config, device="cpu")
        assert len(shard.sharded_state_dict) > 0

    def test_rejects_invalid_model_type(self) -> None:
        """TensorParallelShard raises TypeError for unsupported model types."""
        config = _make_qwen_config(rank=0)
        with pytest.raises(TypeError, match="must be a state dict"):
            TensorParallelShard("not_a_model", config, device="cpu")  # type: ignore[arg-type]

    def test_stores_config_and_device(self) -> None:
        """TensorParallelShard stores config and device attributes."""
        config = _make_qwen_config(rank=2)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert shard.config is config
        assert shard.device == "cpu"


class TestShardWeightsQKV:
    """Tests for QKV projection sharding (column-parallel, output dim)."""

    def test_q_proj_shape_rank0(self) -> None:
        """q_proj is sliced along output dim: (heads_per_rank * head_dim, hidden_size)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # q_proj: full (2560, 2560) → per-rank (640, 2560) with 8 heads * 80 head_dim
        q_weight = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        assert q_weight.shape == (640, 2560)

    def test_k_proj_shape_rank0(self) -> None:
        """k_proj is sliced along output dim: (kv_heads_per_rank * head_dim, hidden_size)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # k_proj: full (640, 2560) → per-rank (160, 2560) with 2 kv_heads * 80 head_dim
        k_weight = shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"]
        assert k_weight.shape == (160, 2560)

    def test_v_proj_shape_rank0(self) -> None:
        """v_proj is sliced along output dim: (kv_heads_per_rank * head_dim, hidden_size)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # v_proj: full (640, 2560) → per-rank (160, 2560)
        v_weight = shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"]
        assert v_weight.shape == (160, 2560)

    def test_qkv_different_ranks_get_different_slices(self) -> None:
        """Different ranks get non-overlapping slices of q_proj."""
        state_dict = _make_fake_state_dict(num_layers=1)
        shards = []
        for rank in range(4):
            config = _make_qwen_config(rank=rank, world_size=4)
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shards.append(shard)

        # Concatenating all rank slices should reconstruct the original
        q_slices = [
            s.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
            for s in shards
        ]
        reconstructed = torch.cat(q_slices, dim=0)
        original = state_dict["model.layers.0.self_attn.q_proj.weight"]
        assert torch.allclose(reconstructed, original)


class TestShardWeightsAttentionOutput:
    """Tests for attention output projection sharding (row-parallel, input dim)."""

    def test_o_proj_shape_rank0(self) -> None:
        """o_proj is sliced along input dim: (hidden_size, heads_per_rank * head_dim)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # o_proj: full (2560, 2560) → per-rank (2560, 640)
        o_weight = shard.sharded_state_dict["model.layers.0.self_attn.o_proj.weight"]
        assert o_weight.shape == (2560, 640)

    def test_o_proj_reconstruction(self) -> None:
        """Concatenating all rank o_proj slices along dim 1 reconstructs original."""
        state_dict = _make_fake_state_dict(num_layers=1)
        shards = []
        for rank in range(4):
            config = _make_qwen_config(rank=rank, world_size=4)
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shards.append(shard)

        o_slices = [
            s.sharded_state_dict["model.layers.0.self_attn.o_proj.weight"]
            for s in shards
        ]
        reconstructed = torch.cat(o_slices, dim=1)
        original = state_dict["model.layers.0.self_attn.o_proj.weight"]
        assert torch.allclose(reconstructed, original)


class TestShardWeightsMLP:
    """Tests for MLP projection sharding."""

    def test_gate_proj_shape(self) -> None:
        """gate_proj is column-parallel: (intermediate_per_rank, hidden_size)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # gate_proj: full (6912, 2560) → per-rank (1728, 2560)
        gate_weight = shard.sharded_state_dict["model.layers.0.mlp.gate_proj.weight"]
        assert gate_weight.shape == (1728, 2560)

    def test_up_proj_shape(self) -> None:
        """up_proj is column-parallel: (intermediate_per_rank, hidden_size)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # up_proj: full (6912, 2560) → per-rank (1728, 2560)
        up_weight = shard.sharded_state_dict["model.layers.0.mlp.up_proj.weight"]
        assert up_weight.shape == (1728, 2560)

    def test_down_proj_shape(self) -> None:
        """down_proj is row-parallel: (hidden_size, intermediate_per_rank)."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict()
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # down_proj: full (2560, 6912) → per-rank (2560, 1728)
        down_weight = shard.sharded_state_dict["model.layers.0.mlp.down_proj.weight"]
        assert down_weight.shape == (2560, 1728)

    def test_mlp_gate_reconstruction(self) -> None:
        """Concatenating all rank gate_proj slices along dim 0 reconstructs original."""
        state_dict = _make_fake_state_dict(num_layers=1)
        shards = []
        for rank in range(4):
            config = _make_qwen_config(rank=rank, world_size=4)
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shards.append(shard)

        gate_slices = [
            s.sharded_state_dict["model.layers.0.mlp.gate_proj.weight"]
            for s in shards
        ]
        reconstructed = torch.cat(gate_slices, dim=0)
        original = state_dict["model.layers.0.mlp.gate_proj.weight"]
        assert torch.allclose(reconstructed, original)

    def test_mlp_down_reconstruction(self) -> None:
        """Concatenating all rank down_proj slices along dim 1 reconstructs original."""
        state_dict = _make_fake_state_dict(num_layers=1)
        shards = []
        for rank in range(4):
            config = _make_qwen_config(rank=rank, world_size=4)
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shards.append(shard)

        down_slices = [
            s.sharded_state_dict["model.layers.0.mlp.down_proj.weight"]
            for s in shards
        ]
        reconstructed = torch.cat(down_slices, dim=1)
        original = state_dict["model.layers.0.mlp.down_proj.weight"]
        assert torch.allclose(reconstructed, original)


class TestRedundantParameters:
    """Tests for parameters that should NOT be sharded."""

    def test_embed_tokens_not_sharded(self) -> None:
        """Embedding table is kept redundant (full copy on each rank)."""
        state_dict = _make_fake_state_dict(num_layers=1)
        original_shape = state_dict["model.embed_tokens.weight"].shape

        for rank in range(4):
            config = _make_qwen_config(rank=rank, world_size=4)
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            embed = shard.sharded_state_dict["model.embed_tokens.weight"]
            assert embed.shape == original_shape

    def test_lm_head_not_sharded(self) -> None:
        """lm_head is kept redundant (full copy on each rank)."""
        state_dict = _make_fake_state_dict(num_layers=1)
        original_shape = state_dict["lm_head.weight"].shape

        config = _make_qwen_config(rank=0, world_size=4)
        shard = TensorParallelShard(dict(state_dict), config, device="cpu")
        lm_head = shard.sharded_state_dict["lm_head.weight"]
        assert lm_head.shape == original_shape

    def test_layer_norms_not_sharded(self) -> None:
        """Layer norms are kept redundant."""
        state_dict = _make_fake_state_dict(num_layers=1)
        config = _make_qwen_config(rank=2, world_size=4)
        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        input_ln = shard.sharded_state_dict["model.layers.0.input_layernorm.weight"]
        post_ln = shard.sharded_state_dict["model.layers.0.post_attention_layernorm.weight"]
        final_norm = shard.sharded_state_dict["model.norm.weight"]

        assert input_ln.shape == (2560,)
        assert post_ln.shape == (2560,)
        assert final_norm.shape == (2560,)


class TestMemoryReduction:
    """Tests for memory reduction from sharding."""

    def test_sharded_memory_less_than_full(self) -> None:
        """Sharded weights use less memory than full model for parallelized layers."""
        state_dict = _make_fake_state_dict(num_layers=2)
        config = _make_qwen_config(rank=0, world_size=4)
        shard = TensorParallelShard(dict(state_dict), config, device="cpu")

        # Calculate memory for sharded attention/MLP params vs full
        full_attn_mlp_bytes = 0
        sharded_attn_mlp_bytes = 0

        for name, tensor in state_dict.items():
            if any(
                key in name
                for key in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]
            ):
                full_attn_mlp_bytes += tensor.numel() * tensor.element_size()

        for name, tensor in shard.sharded_state_dict.items():
            if any(
                key in name
                for key in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]
            ):
                sharded_attn_mlp_bytes += tensor.numel() * tensor.element_size()

        # Sharded should be approximately 1/world_size of full
        ratio = sharded_attn_mlp_bytes / full_attn_mlp_bytes
        assert 0.2 < ratio < 0.3  # ~0.25 for world_size=4


class TestKVCacheShape:
    """Tests for KV cache shape matching head-parallel assignment.

    Requirements: 3.5, 6.5
    """

    def test_kv_cache_shape_matches_heads_per_rank(self) -> None:
        """KV cache shape is [batch, kv_heads_per_rank, seq_len, head_dim].

        The forward pass maintains a head-parallel KV cache where each rank
        stores only its assigned KV heads.
        """
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Mock _all_reduce to be a no-op (single rank scenario)
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        # Run forward pass with token IDs
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        _, kv_cache = shard.forward(input_ids)

        # KV cache should have one entry per layer
        assert len(kv_cache) == 1

        # Each entry is (key, value) tuple
        key_cache, value_cache = kv_cache[0]

        # Shape: [batch, kv_heads_per_rank, seq_len, head_dim]
        # For Qwen3.5-4B with world_size=4: kv_heads_per_rank = 8/4 = 2
        expected_shape = (batch_size, config.kv_heads_per_rank, seq_len, config.head_dim)
        assert key_cache.shape == expected_shape, (
            f"Key cache shape {key_cache.shape} != expected {expected_shape}"
        )
        assert value_cache.shape == expected_shape, (
            f"Value cache shape {value_cache.shape} != expected {expected_shape}"
        )

    def test_kv_cache_grows_with_decode_steps(self) -> None:
        """KV cache seq_len dimension grows as tokens are decoded."""
        config = _make_qwen_config(rank=1, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Mock _all_reduce to be a no-op
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        # Prefill with 3 tokens
        input_ids = torch.randint(0, 100, (1, 3))
        _, kv_cache = shard.forward(input_ids)

        key_cache, value_cache = kv_cache[0]
        assert key_cache.shape[2] == 3  # seq_len = 3

        # Decode one more token using the KV cache
        next_token = torch.randint(0, 100, (1, 1))
        _, kv_cache_2 = shard.forward(next_token, past_key_values=kv_cache)

        key_cache_2, value_cache_2 = kv_cache_2[0]
        assert key_cache_2.shape[2] == 4  # seq_len grew to 4
        assert value_cache_2.shape[2] == 4

    def test_kv_cache_different_world_sizes(self) -> None:
        """KV cache kv_heads dimension scales with world_size."""
        state_dict = _make_fake_state_dict(num_layers=1)

        for world_size in [2, 4]:
            config = TPShardConfig(
                rank=0,
                world_size=world_size,
                hidden_size=2560,
                num_attention_heads=32,
                head_dim=80,
                intermediate_size=6912,
                num_key_value_heads=8,
            )
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

            input_ids = torch.randint(0, 100, (1, 2))
            _, kv_cache = shard.forward(input_ids)

            key_cache, _ = kv_cache[0]
            expected_kv_heads = 8 // world_size
            assert key_cache.shape[1] == expected_kv_heads, (
                f"world_size={world_size}: kv_heads {key_cache.shape[1]} != {expected_kv_heads}"
            )


class TestColumnParallelLinear:
    """Tests for _column_parallel_linear() method."""

    def test_produces_correct_output_shape(self) -> None:
        """Column-parallel linear produces output with shard's output dim."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Simulate input: [batch=1, seq_len=3, hidden_size=2560]
        input_tensor = torch.randn(1, 3, 2560)
        # Column-parallel weight shard: (intermediate_per_rank, hidden_size) = (1728, 2560)
        weight = torch.randn(1728, 2560)

        result = shard._column_parallel_linear(input_tensor, weight)
        assert result.shape == (1, 3, 1728)

    def test_with_bias(self) -> None:
        """Column-parallel linear applies bias correctly."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        input_tensor = torch.randn(1, 2, 2560)
        weight = torch.randn(1728, 2560)
        bias = torch.randn(1728)

        result_with_bias = shard._column_parallel_linear(input_tensor, weight, bias)
        result_no_bias = shard._column_parallel_linear(input_tensor, weight, None)

        # Result with bias should differ from result without bias
        assert not torch.allclose(result_with_bias, result_no_bias)
        # The difference should be approximately the bias (floating-point tolerance)
        diff = result_with_bias - result_no_bias
        expected_bias = bias.unsqueeze(0).unsqueeze(0).expand_as(diff)
        assert torch.allclose(diff, expected_bias, atol=1e-5, rtol=1e-5)

    def test_equivalent_to_f_linear(self) -> None:
        """Column-parallel linear is equivalent to F.linear (no communication)."""
        import torch.nn.functional as F

        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        input_tensor = torch.randn(1, 5, 2560)
        weight = torch.randn(640, 2560)
        bias = torch.randn(640)

        result = shard._column_parallel_linear(input_tensor, weight, bias)
        expected = F.linear(input_tensor, weight, bias)
        assert torch.allclose(result, expected)


class TestRowParallelLinear:
    """Tests for _row_parallel_linear() method."""

    def test_produces_correct_output_shape(self) -> None:
        """Row-parallel linear produces output with full hidden_size dim."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Simulate input: [batch=1, seq_len=3, intermediate_per_rank=1728]
        input_tensor = torch.randn(1, 3, 1728)
        # Row-parallel weight shard: (hidden_size, intermediate_per_rank) = (2560, 1728)
        weight = torch.randn(2560, 1728)

        # Mock the all_reduce to be a no-op (single rank scenario)
        original_all_reduce = shard._all_reduce
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        result = shard._row_parallel_linear(input_tensor, weight)
        assert result.shape == (1, 3, 2560)

        shard._all_reduce = original_all_reduce  # type: ignore[assignment]

    def test_bias_applied_after_all_reduce(self) -> None:
        """Row-parallel linear applies bias AFTER all-reduce, not before."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        input_tensor = torch.randn(1, 2, 1728)
        weight = torch.randn(2560, 1728)
        bias = torch.randn(2560)

        # Track what tensor goes through all_reduce
        all_reduce_input: list[torch.Tensor] = []

        def mock_all_reduce(tensor: torch.Tensor, layer_index: int = -1) -> torch.Tensor:
            all_reduce_input.append(tensor.clone())
            return tensor

        shard._all_reduce = mock_all_reduce  # type: ignore[assignment]

        result = shard._row_parallel_linear(input_tensor, weight, bias, layer_index=0)

        # The tensor passed to all_reduce should NOT include bias
        import torch.nn.functional as F

        expected_partial = F.linear(input_tensor, weight, None)
        assert torch.allclose(all_reduce_input[0], expected_partial)

        # Final result should include bias
        expected_result = expected_partial + bias
        assert torch.allclose(result, expected_result)

    def test_passes_layer_index_to_all_reduce(self) -> None:
        """Row-parallel linear passes layer_index to _all_reduce for error context."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        input_tensor = torch.randn(1, 1, 1728)
        weight = torch.randn(2560, 1728)

        captured_layer_index: list[int] = []

        def mock_all_reduce(tensor: torch.Tensor, layer_index: int = -1) -> torch.Tensor:
            captured_layer_index.append(layer_index)
            return tensor

        shard._all_reduce = mock_all_reduce  # type: ignore[assignment]

        shard._row_parallel_linear(input_tensor, weight, layer_index=7)
        assert captured_layer_index[0] == 7


class TestAllReduce:
    """Tests for _all_reduce() method."""

    def test_raises_runtime_error_on_failure(self) -> None:
        """_all_reduce raises RuntimeError with context when dist.all_reduce fails."""
        config = _make_qwen_config(rank=2, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        tensor = torch.randn(1, 3, 2560)

        # Without a process group initialized, all_reduce should fail
        # We test that the error includes the expected context
        with pytest.raises(RuntimeError, match="layer_index=5"):
            shard._all_reduce(tensor, layer_index=5)

    def test_error_includes_tensor_shape(self) -> None:
        """RuntimeError from _all_reduce includes the tensor shape."""
        config = _make_qwen_config(rank=1, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        tensor = torch.randn(2, 4, 2560)

        with pytest.raises(RuntimeError, match=r"tensor_shape=\(2, 4, 2560\)"):
            shard._all_reduce(tensor, layer_index=0)

    def test_error_includes_timeout_and_rank(self) -> None:
        """RuntimeError from _all_reduce includes timeout value and rank."""
        config = _make_qwen_config(rank=3, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        tensor = torch.randn(1, 1, 2560)

        with pytest.raises(RuntimeError) as exc_info:
            shard._all_reduce(tensor, layer_index=10)

        error_msg = str(exc_info.value)
        assert "timeout=30s" in error_msg
        assert "rank=3" in error_msg

# ============================================================================
# Tests for Task 1: Architecture Detection, Fused QKV, Key Consistency, Errors
# ============================================================================

from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import ModelArchitecture


def _make_phi_config(rank: int = 0, world_size: int = 2) -> TPShardConfig:
    """Create a Phi-4-like TPShardConfig.

    Uses world_size=2 by default because Phi-4 has 10 KV heads
    which is divisible by 2 but not by 4.
    """
    return TPShardConfig(
        rank=rank,
        world_size=world_size,
        hidden_size=6144,
        num_attention_heads=40,
        head_dim=128,
        intermediate_size=16384,
        num_key_value_heads=10,
    )


def _make_phi_state_dict_fused(
    hidden_size: int = 6144,
    num_heads: int = 40,
    head_dim: int = 128,
    num_kv_heads: int = 10,
    intermediate_size: int = 16384,
    num_layers: int = 2,
) -> dict[str, torch.Tensor]:
    """Create a fake Phi-style state dict with fused QKV and LayerNorm biases."""
    state_dict: dict[str, torch.Tensor] = {}

    # Embedding and final layers (redundant)
    state_dict["model.embed_tokens.weight"] = torch.randn(32064, hidden_size)
    state_dict["model.norm.weight"] = torch.randn(hidden_size)
    state_dict["model.norm.bias"] = torch.randn(hidden_size)
    state_dict["lm_head.weight"] = torch.randn(32064, hidden_size)

    # Fused QKV size: (num_heads + 2 * num_kv_heads) * head_dim
    fused_qkv_size = (num_heads + 2 * num_kv_heads) * head_dim

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Layer norms with bias (LayerNorm, Phi-style)
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.input_layernorm.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.post_attention_layernorm.bias"] = torch.randn(hidden_size)

        # Fused QKV projection
        state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = torch.randn(
            fused_qkv_size, hidden_size
        )
        state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = torch.randn(fused_qkv_size)

        # Output projection
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden_size, num_heads * head_dim
        )
        state_dict[f"{prefix}.self_attn.o_proj.bias"] = torch.randn(hidden_size)

        # MLP projections
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


def _make_qwen_state_dict_with_separate_qkv(
    hidden_size: int = 2560,
    num_heads: int = 32,
    head_dim: int = 80,
    num_kv_heads: int = 8,
    intermediate_size: int = 6912,
    num_layers: int = 2,
) -> dict[str, torch.Tensor]:
    """Create a Qwen/Llama-style state dict with separate Q, K, V and RMSNorm."""
    return _make_fake_state_dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
    )


class TestArchitectureDetection:
    """Tests for _detect_architecture() method (Task 1.1)."""

    def test_detects_qwen_llama_from_separate_qkv(self) -> None:
        """Qwen/Llama architecture detected when separate q/k/v_proj keys exist."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_qwen_state_dict_with_separate_qkv(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert shard.architecture == ModelArchitecture.QWEN_LLAMA

    def test_detects_phi_from_layernorm_bias(self) -> None:
        """Phi architecture detected when input_layernorm.bias is present."""
        config = _make_phi_config(rank=0)
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert shard.architecture == ModelArchitecture.PHI

    def test_detects_qwen_llama_without_bias(self) -> None:
        """Qwen/Llama detected when no layernorm bias exists."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        # Verify no bias keys exist
        assert not any("layernorm.bias" in k for k in state_dict)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert shard.architecture == ModelArchitecture.QWEN_LLAMA

    def test_architecture_is_enum_value(self) -> None:
        """Architecture detection returns a ModelArchitecture enum member."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert isinstance(shard.architecture, ModelArchitecture)

    def test_phi_detection_with_multiple_layers(self) -> None:
        """Phi detection works with multiple layers."""
        config = _make_phi_config(rank=0)
        state_dict = _make_phi_state_dict_fused(num_layers=3)
        shard = TensorParallelShard(state_dict, config, device="cpu")
        assert shard.architecture == ModelArchitecture.PHI


class TestFusedQKVSplitting:
    """Tests for fused QKV weight splitting (Task 1.2)."""

    def test_fused_qkv_produces_separate_keys(self) -> None:
        """Fused qkv_proj.weight is split into q_proj, k_proj, v_proj keys."""
        config = _make_phi_config(rank=0)
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # After splitting, separate keys exist
        assert "model.layers.0.self_attn.q_proj.weight" in shard.sharded_state_dict
        assert "model.layers.0.self_attn.k_proj.weight" in shard.sharded_state_dict
        assert "model.layers.0.self_attn.v_proj.weight" in shard.sharded_state_dict

        # Original fused key does NOT exist
        assert "model.layers.0.self_attn.qkv_proj.weight" not in shard.sharded_state_dict

    def test_fused_qkv_bias_produces_separate_bias_keys(self) -> None:
        """Fused qkv_proj.bias is split into q_proj, k_proj, v_proj bias keys."""
        config = _make_phi_config(rank=0)
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        assert "model.layers.0.self_attn.q_proj.bias" in shard.sharded_state_dict
        assert "model.layers.0.self_attn.k_proj.bias" in shard.sharded_state_dict
        assert "model.layers.0.self_attn.v_proj.bias" in shard.sharded_state_dict
        assert "model.layers.0.self_attn.qkv_proj.bias" not in shard.sharded_state_dict

    def test_fused_qkv_split_shapes_correct(self) -> None:
        """Split Q, K, V shards have correct shapes for rank 0 with world_size=2."""
        config = _make_phi_config(rank=0, world_size=2)
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Phi-4: 40 heads, 10 kv_heads, head_dim=128, world_size=2
        # heads_per_rank = 40/2 = 20, kv_heads_per_rank = 10/2 = 5
        # q_shard: (20 * 128, 6144) = (2560, 6144)
        # k_shard: (5 * 128, 6144) = (640, 6144)
        # v_shard: (5 * 128, 6144) = (640, 6144)
        q_weight = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        k_weight = shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"]
        v_weight = shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"]

        assert q_weight.shape == (2560, 6144)
        assert k_weight.shape == (640, 6144)
        assert v_weight.shape == (640, 6144)

    def test_fused_qkv_split_shapes_world_size_2(self) -> None:
        """Split Q, K, V shards have correct shapes with world_size=2."""
        config = TPShardConfig(
            rank=0,
            world_size=2,
            hidden_size=6144,
            num_attention_heads=40,
            head_dim=128,
            intermediate_size=16384,
            num_key_value_heads=10,
        )
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # heads_per_rank = 40/2 = 20, kv_heads_per_rank = 10/2 = 5
        # q_shard: (20 * 128, 6144) = (2560, 6144)
        # k_shard: (5 * 128, 6144) = (640, 6144)
        # v_shard: (5 * 128, 6144) = (640, 6144)
        q_weight = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        k_weight = shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"]
        v_weight = shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"]

        assert q_weight.shape == (2560, 6144)
        assert k_weight.shape == (640, 6144)
        assert v_weight.shape == (640, 6144)

    def test_fused_qkv_split_bias_shapes(self) -> None:
        """Split Q, K, V bias shards have correct 1D shapes."""
        config = TPShardConfig(
            rank=0,
            world_size=2,
            hidden_size=6144,
            num_attention_heads=40,
            head_dim=128,
            intermediate_size=16384,
            num_key_value_heads=10,
        )
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        q_bias = shard.sharded_state_dict["model.layers.0.self_attn.q_proj.bias"]
        k_bias = shard.sharded_state_dict["model.layers.0.self_attn.k_proj.bias"]
        v_bias = shard.sharded_state_dict["model.layers.0.self_attn.v_proj.bias"]

        assert q_bias.shape == (2560,)
        assert k_bias.shape == (640,)
        assert v_bias.shape == (640,)

    def test_fused_qkv_round_trip_reconstruction(self) -> None:
        """Concatenating Q, K, V shards from all ranks reconstructs the original fused weight."""
        world_size = 2
        hidden_size = 6144
        num_heads = 40
        head_dim = 128
        num_kv_heads = 10
        intermediate_size = 16384

        state_dict = _make_phi_state_dict_fused(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            num_layers=1,
        )
        original_fused = state_dict["model.layers.0.self_attn.qkv_proj.weight"].clone()

        # Shard across all ranks
        q_shards = []
        k_shards = []
        v_shards = []
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
            q_shards.append(shard.sharded_state_dict["model.layers.0.self_attn.q_proj.weight"])
            k_shards.append(shard.sharded_state_dict["model.layers.0.self_attn.k_proj.weight"])
            v_shards.append(shard.sharded_state_dict["model.layers.0.self_attn.v_proj.weight"])

        # Reconstruct full Q, K, V
        q_full = torch.cat(q_shards, dim=0)
        k_full = torch.cat(k_shards, dim=0)
        v_full = torch.cat(v_shards, dim=0)

        # Reconstruct fused
        reconstructed = torch.cat([q_full, k_full, v_full], dim=0)
        assert torch.allclose(reconstructed, original_fused)

    def test_fused_qkv_different_ranks_get_different_slices(self) -> None:
        """Different ranks get non-overlapping slices of the fused QKV."""
        world_size = 2
        state_dict = _make_phi_state_dict_fused(num_layers=1)

        shards = []
        for rank in range(world_size):
            config = TPShardConfig(
                rank=rank,
                world_size=world_size,
                hidden_size=6144,
                num_attention_heads=40,
                head_dim=128,
                intermediate_size=16384,
                num_key_value_heads=10,
            )
            shard = TensorParallelShard(dict(state_dict), config, device="cpu")
            shards.append(shard)

        # Rank 0 and rank 1 q_proj should be different
        q0 = shards[0].sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        q1 = shards[1].sharded_state_dict["model.layers.0.self_attn.q_proj.weight"]
        assert not torch.allclose(q0, q1)


class TestKeyConsistency:
    """Tests for key consistency after shard_weights() (Task 1.5).

    Verifies that forward() can access all needed keys without KeyError.
    """

    def test_qwen_forward_no_key_error(self) -> None:
        """Forward pass on Qwen state dict does not raise KeyError."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=2)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Mock _all_reduce to be a no-op
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        input_ids = torch.randint(0, 100, (1, 3))
        # This should not raise KeyError
        logits, kv_cache = shard.forward(input_ids)
        assert logits.shape[0] == 1
        assert logits.shape[1] == 3
        assert len(kv_cache) == 2

    def test_phi_forward_no_key_error(self) -> None:
        """Forward pass on Phi state dict (fused QKV split) does not raise KeyError."""
        config = TPShardConfig(
            rank=0,
            world_size=2,
            hidden_size=6144,
            num_attention_heads=40,
            head_dim=128,
            intermediate_size=16384,
            num_key_value_heads=10,
        )
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Mock _all_reduce to be a no-op
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        input_ids = torch.randint(0, 100, (1, 3))
        # This should not raise KeyError — fused QKV was split into separate keys
        logits, kv_cache = shard.forward(input_ids)
        assert logits.shape[0] == 1
        assert logits.shape[1] == 3
        assert len(kv_cache) == 1

    def test_all_forward_keys_exist_in_sharded_dict_qwen(self) -> None:
        """Every key accessed by forward() exists in sharded_state_dict for Qwen."""
        config = _make_qwen_config(rank=0, world_size=4)
        state_dict = _make_fake_state_dict(num_layers=2)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Check all keys that forward() would access
        num_layers = shard._detect_num_layers()
        for layer_idx in range(num_layers):
            assert f"model.layers.{layer_idx}.input_layernorm.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.q_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.k_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.v_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.o_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.post_attention_layernorm.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.gate_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.up_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.down_proj.weight" in shard.sharded_state_dict

        assert "model.embed_tokens.weight" in shard.sharded_state_dict
        assert "model.norm.weight" in shard.sharded_state_dict
        assert "lm_head.weight" in shard.sharded_state_dict

    def test_all_forward_keys_exist_in_sharded_dict_phi(self) -> None:
        """Every key accessed by forward() exists in sharded_state_dict for Phi (fused QKV)."""
        config = TPShardConfig(
            rank=0,
            world_size=2,
            hidden_size=6144,
            num_attention_heads=40,
            head_dim=128,
            intermediate_size=16384,
            num_key_value_heads=10,
        )
        state_dict = _make_phi_state_dict_fused(num_layers=2)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        num_layers = shard._detect_num_layers()
        for layer_idx in range(num_layers):
            # After fused QKV split, separate keys exist
            assert f"model.layers.{layer_idx}.self_attn.q_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.k_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.v_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.self_attn.o_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.gate_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.up_proj.weight" in shard.sharded_state_dict
            assert f"model.layers.{layer_idx}.mlp.down_proj.weight" in shard.sharded_state_dict

        assert "model.embed_tokens.weight" in shard.sharded_state_dict
        assert "model.norm.weight" in shard.sharded_state_dict
        assert "lm_head.weight" in shard.sharded_state_dict


class TestDescriptiveKeyError:
    """Tests for descriptive KeyError in _get_weight() (Task 1.3)."""

    def test_missing_key_raises_key_error(self) -> None:
        """_get_weight raises KeyError for missing keys."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        with pytest.raises(KeyError):
            shard._get_weight("model.layers.99.self_attn.q_proj.weight")

    def test_error_includes_missing_key_name(self) -> None:
        """KeyError message includes the missing key name."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        with pytest.raises(KeyError, match="model.layers.99.self_attn.q_proj.weight"):
            shard._get_weight("model.layers.99.self_attn.q_proj.weight")

    def test_error_includes_similar_keys(self) -> None:
        """KeyError message includes available keys with the same prefix."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Request a key with a valid layer prefix but wrong suffix
        with pytest.raises(KeyError) as exc_info:
            shard._get_weight("model.layers.0.self_attn.nonexistent_proj.weight")

        error_msg = str(exc_info.value)
        # The prefix is "model.layers.0.self_attn.nonexistent_proj"
        # Available keys with that prefix should be empty, but the message
        # should still contain the prefix info
        assert "model.layers.0.self_attn.nonexistent_proj" in error_msg

    def test_error_lists_keys_with_matching_prefix(self) -> None:
        """KeyError lists keys that share the same layer prefix."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Request a key where the prefix matches existing keys
        with pytest.raises(KeyError) as exc_info:
            shard._get_weight("model.layers.0.self_attn.missing")

        error_msg = str(exc_info.value)
        # The prefix is "model.layers.0.self_attn" which has real keys
        assert "model.layers.0.self_attn" in error_msg
        # Should list available keys with that prefix
        assert "q_proj.weight" in error_msg

    def test_get_weight_optional_returns_none_for_missing(self) -> None:
        """_get_weight_optional returns None for missing keys without error."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        result = shard._get_weight_optional("model.layers.0.self_attn.q_proj.bias")
        assert result is None


class TestLayerNormSupport:
    """Tests for LayerNorm support alongside RMSNorm (Task 1.4)."""

    def test_apply_norm_uses_rms_norm_without_bias(self) -> None:
        """_apply_norm uses RMSNorm when bias is None."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        hidden_states = torch.randn(1, 3, 2560)
        weight = torch.ones(2560)

        # RMSNorm result
        rms_result = shard._rms_norm(hidden_states, weight)
        # _apply_norm with no bias should give same result
        apply_result = shard._apply_norm(hidden_states, weight, bias=None)

        assert torch.allclose(rms_result, apply_result, atol=1e-6)

    def test_apply_norm_uses_layer_norm_with_bias(self) -> None:
        """_apply_norm uses F.layer_norm when bias is provided."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        hidden_states = torch.randn(1, 3, 2560)
        weight = torch.ones(2560)
        bias = torch.zeros(2560)

        # F.layer_norm result
        expected = F.layer_norm(hidden_states, [2560], weight, bias, 1e-6)
        # _apply_norm with bias should give same result
        apply_result = shard._apply_norm(hidden_states, weight, bias=bias, eps=1e-6)

        assert torch.allclose(expected, apply_result, atol=1e-6)

    def test_layer_norm_differs_from_rms_norm(self) -> None:
        """LayerNorm and RMSNorm produce different results for same input."""
        config = _make_qwen_config(rank=0)
        state_dict = _make_fake_state_dict(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        hidden_states = torch.randn(1, 3, 2560)
        weight = torch.ones(2560)
        bias = torch.randn(2560)  # Non-zero bias

        rms_result = shard._apply_norm(hidden_states, weight, bias=None)
        ln_result = shard._apply_norm(hidden_states, weight, bias=bias)

        # Results should differ because LayerNorm subtracts mean and adds bias
        assert not torch.allclose(rms_result, ln_result)

    def test_phi_forward_uses_layer_norm(self) -> None:
        """Forward pass on Phi model uses LayerNorm (bias present in state dict)."""
        config = TPShardConfig(
            rank=0,
            world_size=2,
            hidden_size=6144,
            num_attention_heads=40,
            head_dim=128,
            intermediate_size=16384,
            num_key_value_heads=10,
        )
        state_dict = _make_phi_state_dict_fused(num_layers=1)
        shard = TensorParallelShard(state_dict, config, device="cpu")

        # Mock _all_reduce to be a no-op
        shard._all_reduce = lambda tensor, layer_index=-1: tensor  # type: ignore[assignment]

        # Verify bias keys exist (LayerNorm indicator)
        assert "model.layers.0.input_layernorm.bias" in shard.sharded_state_dict
        assert "model.layers.0.post_attention_layernorm.bias" in shard.sharded_state_dict

        # Forward pass should work without error
        input_ids = torch.randint(0, 100, (1, 2))
        logits, _ = shard.forward(input_ids)
        assert logits.shape == (1, 2, 32064)
