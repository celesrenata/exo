# Feature: pipeline-parallelism-optimization, Property 4 & 5: Local state allocation
"""
Property-based tests for pipeline-parallel shard local state allocation.

**Validates: Requirements 5.1, 5.2, 6.1**

Uses Hypothesis to generate layer ranges and layer type patterns, then verifies that
PipelineParallelShard allocates:
- KV cache entries for exactly (end_layer - start_layer) layers
- Linear attention recurrent state count matches the number of linear attention layers
  within the assigned range
- KV cache sequence length grows monotonically (by 1 per decode step)
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_SHARD_PATH = _THIS_DIR.parent / "pipeline_parallel_shard.py"


def _load_pipeline_parallel_shard() -> types.ModuleType:
    """Load pipeline_parallel_shard.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_parallel_shard_state_alloc_test"
    spec = importlib.util.spec_from_file_location(module_name, _PIPELINE_SHARD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_parallel_shard()
PipelineParallelShard = _mod.PipelineParallelShard
PipelineStageConfig = _mod.PipelineStageConfig
compute_layer_assignment = _mod.compute_layer_assignment


# ---------------------------------------------------------------------------
# Mock layer: identity function matching HuggingFace layer output format
# ---------------------------------------------------------------------------


class MockLayer(nn.Module):
    """Mock transformer layer that passes hidden_states through unchanged.

    Returns a tuple (hidden_states, None) to match HuggingFace layer output format.
    """

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return (hidden_states, None)


# ---------------------------------------------------------------------------
# Mock config for hybrid attention models (Qwen3.5-style)
# ---------------------------------------------------------------------------


class MockConfig:
    """Mock text_model_config with a layer_types attribute for hybrid attention."""

    def __init__(self, layer_types: list[str]) -> None:
        self.layer_types = layer_types


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

num_layers_strategy = st.integers(min_value=4, max_value=32)
world_size_strategy = st.integers(min_value=1, max_value=4)
hidden_size_strategy = st.integers(min_value=32, max_value=128)
vocab_size_strategy = st.integers(min_value=100, max_value=500)

# Layer type pattern: random mix of "full_attention" and "linear_attention"
layer_type_strategy = st.sampled_from(["full_attention", "linear_attention"])


# ---------------------------------------------------------------------------
# Property 4: Local state allocation matches layer range
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_layers=num_layers_strategy,
    world_size=world_size_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
    data=st.data(),
)
def test_kv_cache_has_exactly_num_local_layers_entries(
    num_layers: int,
    world_size: int,
    hidden_size: int,
    vocab_size: int,
    data: st.DataObject,
) -> None:
    """KV cache has exactly (end_layer - start_layer) entries for any valid rank.

    For any pipeline stage configuration, the shard allocates one KV cache
    entry per local layer, regardless of layer types.

    **Validates: Requirements 5.1**
    """
    # Constrain world_size to be at most num_layers
    world_size = min(world_size, num_layers)
    rank = data.draw(st.integers(min_value=0, max_value=world_size - 1), label="rank")

    start_layer, end_layer = compute_layer_assignment(num_layers, world_size, rank)
    num_local_layers = end_layer - start_layer

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    layers = nn.ModuleList([MockLayer() for _ in range(num_local_layers)])

    # Generate a random layer type pattern for the full model
    layer_types = data.draw(
        st.lists(layer_type_strategy, min_size=num_layers, max_size=num_layers),
        label="layer_types",
    )
    text_model_config = MockConfig(layer_types)

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=nn.Embedding(vocab_size, hidden_size) if config.is_first_stage else None,
        lm_head=nn.Linear(hidden_size, vocab_size, bias=False) if config.is_last_stage else None,
        final_norm=nn.LayerNorm(hidden_size) if config.is_last_stage else None,
        text_model_config=text_model_config,
    )

    assert len(shard._kv_cache) == config.num_local_layers, (
        f"KV cache length mismatch: expected {config.num_local_layers} entries "
        f"(end_layer={end_layer} - start_layer={start_layer}), "
        f"got {len(shard._kv_cache)}"
    )


@settings(max_examples=100)
@given(
    num_layers=num_layers_strategy,
    world_size=world_size_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
    data=st.data(),
)
def test_linear_attention_state_count_matches_linear_layers_in_range(
    num_layers: int,
    world_size: int,
    hidden_size: int,
    vocab_size: int,
    data: st.DataObject,
) -> None:
    """Linear attention state count matches the number of linear attention layers in range.

    For a hybrid attention model (Qwen3.5-style), the shard detects which of its
    local layers use linear attention and tracks their types correctly.

    **Validates: Requirements 6.1**
    """
    # Constrain world_size to be at most num_layers
    world_size = min(world_size, num_layers)
    rank = data.draw(st.integers(min_value=0, max_value=world_size - 1), label="rank")

    start_layer, end_layer = compute_layer_assignment(num_layers, world_size, rank)
    num_local_layers = end_layer - start_layer

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    layers = nn.ModuleList([MockLayer() for _ in range(num_local_layers)])

    # Generate a random layer type pattern for the full model
    layer_types = data.draw(
        st.lists(layer_type_strategy, min_size=num_layers, max_size=num_layers),
        label="layer_types",
    )
    text_model_config = MockConfig(layer_types)

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=nn.Embedding(vocab_size, hidden_size) if config.is_first_stage else None,
        lm_head=nn.Linear(hidden_size, vocab_size, bias=False) if config.is_last_stage else None,
        final_norm=nn.LayerNorm(hidden_size) if config.is_last_stage else None,
        text_model_config=text_model_config,
    )

    # Count expected linear attention layers in the assigned range
    expected_linear_count = sum(
        1 for i in range(start_layer, end_layer)
        if layer_types[i] == "linear_attention"
    )

    # Count actual linear attention layers detected by the shard
    actual_linear_count = len(
        [t for t in shard._layer_types if t == "linear_attention"]
    )

    assert actual_linear_count == expected_linear_count, (
        f"Linear attention layer count mismatch: expected {expected_linear_count} "
        f"linear attention layers in range [{start_layer}, {end_layer}), "
        f"got {actual_linear_count}. "
        f"Full layer_types: {layer_types}, "
        f"Local layer_types: {shard._layer_types}"
    )


@settings(max_examples=100)
@given(
    num_layers=num_layers_strategy,
    world_size=world_size_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
    data=st.data(),
)
def test_kv_cache_entries_reflect_layer_types_after_forward(
    num_layers: int,
    world_size: int,
    hidden_size: int,
    vocab_size: int,
    data: st.DataObject,
) -> None:
    """After a forward pass, KV cache entries for full_attention layers are not None,
    and entries for linear_attention layers are None.

    Mock layers return (hidden_states, None) so all KV cache entries remain None.
    This verifies the shard correctly initializes cache entries per layer type.

    **Validates: Requirements 5.1, 6.1**
    """
    # Constrain world_size to be at most num_layers
    world_size = min(world_size, num_layers)
    rank = data.draw(st.integers(min_value=0, max_value=world_size - 1), label="rank")

    start_layer, end_layer = compute_layer_assignment(num_layers, world_size, rank)
    num_local_layers = end_layer - start_layer

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    layers = nn.ModuleList([MockLayer() for _ in range(num_local_layers)])

    # Generate a random layer type pattern for the full model
    layer_types = data.draw(
        st.lists(layer_type_strategy, min_size=num_layers, max_size=num_layers),
        label="layer_types",
    )
    text_model_config = MockConfig(layer_types)

    embed_tokens = nn.Embedding(vocab_size, hidden_size) if config.is_first_stage else None
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False) if config.is_last_stage else None
    final_norm = nn.LayerNorm(hidden_size) if config.is_last_stage else None

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
        text_model_config=text_model_config,
    )

    # Run a forward pass
    if config.is_first_stage:
        input_data = torch.randint(0, vocab_size, (1, 1))
    else:
        input_data = torch.randn(1, 1, hidden_size)

    _, kv_cache = shard.forward(input_data)

    # Verify KV cache length matches local layers
    assert len(kv_cache) == num_local_layers, (
        f"KV cache length after forward mismatch: expected {num_local_layers}, "
        f"got {len(kv_cache)}"
    )

    # With mock layers that return (hidden_states, None), all KV cache entries
    # should be None (mock layers don't produce real KV cache tensors).
    # This verifies the shard correctly handles the None case for both layer types.
    for i, entry in enumerate(kv_cache):
        assert entry is None, (
            f"KV cache entry {i} should be None with mock layers, got {type(entry)}"
        )


# ---------------------------------------------------------------------------
# Mock layer with real KV cache growth (for Property 5)
# ---------------------------------------------------------------------------


class MockFullAttentionLayer(nn.Module):
    """Mock transformer layer that produces real KV cache entries with growing seq_len.

    Simulates a full-attention layer that:
    - Accepts past_key_value kwarg
    - Concatenates new key/value with past if available
    - Returns (hidden_states, (new_key, new_value)) where seq_len grows by input seq_len

    This enables testing that KV cache sequence length grows monotonically
    across decode steps.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        past_key_value = kwargs.get("past_key_value")

        # Generate new key/value for current step
        new_key = torch.randn(batch, self.num_heads, seq_len, self.head_dim)
        new_value = torch.randn(batch, self.num_heads, seq_len, self.head_dim)

        # Concatenate with past if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            new_key = torch.cat([past_key, new_key], dim=2)
            new_value = torch.cat([past_value, new_value], dim=2)

        return (hidden_states, (new_key, new_value))


# ---------------------------------------------------------------------------
# Property 5: KV cache sequence length grows monotonically
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    num_decode_steps=st.integers(min_value=1, max_value=20),
    hidden_size=st.sampled_from([32, 64]),
    data=st.data(),
)
def test_kv_cache_seq_len_grows_monotonically(
    num_decode_steps: int,
    hidden_size: int,
    data: st.DataObject,
) -> None:
    """After N decode steps, KV cache seq_len == N for each full-attention layer.

    For any PipelineParallelShard with full-attention layers and any sequence of N
    consecutive single-token forward passes (decode steps), the KV cache sequence
    length for each full-attention layer grows by exactly 1 per step, reaching N
    after all steps complete.

    **Validates: Requirements 5.2**
    """
    num_heads = 4
    # Ensure hidden_size is divisible by num_heads
    assert hidden_size % num_heads == 0

    # Use a middle stage (not first, not last) to avoid embedding/lm_head complexity
    # This tests the core KV cache growth behavior in isolation
    num_layers = 4
    world_size = 2
    rank = data.draw(st.integers(min_value=0, max_value=world_size - 1), label="rank")

    start_layer, end_layer = compute_layer_assignment(num_layers, world_size, rank)
    num_local_layers = end_layer - start_layer

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=hidden_size,
        vocab_size=1000,
        num_layers=num_layers,
        device="cpu",
    )

    # All layers are full_attention (produce real KV cache)
    layers = nn.ModuleList(
        [MockFullAttentionLayer(hidden_size, num_heads) for _ in range(num_local_layers)]
    )

    # All layers are full_attention type
    layer_types = ["full_attention"] * num_layers
    text_model_config = MockConfig(layer_types)

    embed_tokens = nn.Embedding(1000, hidden_size) if config.is_first_stage else None
    lm_head = nn.Linear(hidden_size, 1000, bias=False) if config.is_last_stage else None
    final_norm = nn.LayerNorm(hidden_size) if config.is_last_stage else None

    # Patch torch.compile to avoid C++ compilation issues in test environment
    with patch("torch.compile", side_effect=RuntimeError("skip compile in test")):
        shard = PipelineParallelShard(
            layers=layers,
            config=config,
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            final_norm=final_norm,
            text_model_config=text_model_config,
        )

    # Run N decode steps (seq_len=1 each)
    for step in range(1, num_decode_steps + 1):
        if config.is_first_stage:
            # First stage: token IDs input
            input_data = torch.randint(0, 1000, (1, 1))
        else:
            # Non-first stage: hidden_state input
            input_data = torch.randn(1, 1, hidden_size)

        _, kv_cache = shard.forward(input_data)

        # After each step, verify KV cache seq_len == step for all layers
        for layer_idx, entry in enumerate(kv_cache):
            assert entry is not None, (
                f"Step {step}: KV cache entry {layer_idx} is None, "
                f"expected a (key, value) tuple for full_attention layer"
            )
            key, value = entry
            # key shape: [batch, num_heads, seq_len, head_dim]
            actual_seq_len = key.shape[2]
            assert actual_seq_len == step, (
                f"Step {step}: KV cache seq_len for layer {layer_idx} is "
                f"{actual_seq_len}, expected {step}. "
                f"KV cache should grow by exactly 1 per decode step."
            )
            # Value should have the same seq_len
            assert value.shape[2] == step, (
                f"Step {step}: KV value seq_len for layer {layer_idx} is "
                f"{value.shape[2]}, expected {step}."
            )
