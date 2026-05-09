# Feature: pipeline-parallelism-optimization, Property 2: Forward pass output shape depends on stage position
"""
Property-based tests for pipeline-parallel shard forward pass output shapes.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Uses Hypothesis to generate batch_size (fixed at 1), seq_len (1..32),
hidden_size (32..128), and vocab_size (100..500), then verifies that
PipelineParallelShard produces the correct output shape depending on
whether it is the first, middle, or last pipeline stage.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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
    module_name = "pipeline_parallel_shard_shape_test"
    spec = importlib.util.spec_from_file_location(module_name, _PIPELINE_SHARD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_pipeline_parallel_shard()
PipelineParallelShard = _mod.PipelineParallelShard
PipelineStageConfig = _mod.PipelineStageConfig


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
# Hypothesis strategies
# ---------------------------------------------------------------------------

seq_len_strategy = st.integers(min_value=1, max_value=32)
hidden_size_strategy = st.integers(min_value=32, max_value=128)
vocab_size_strategy = st.integers(min_value=100, max_value=500)


# ---------------------------------------------------------------------------
# Property 2: Forward pass output shape depends on stage position
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    seq_len=seq_len_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
)
def test_first_stage_accepts_token_ids_outputs_hidden_state(
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """First stage (rank 0) accepts [1, seq_len] token IDs, outputs [1, seq_len, hidden_size].

    When world_size > 1 and rank == 0, the shard embeds token IDs and runs through
    local layers, producing a hidden_state tensor (not logits).

    **Validates: Requirements 2.1, 2.2**
    """
    num_layers = 4
    world_size = 2  # At least 2 stages so rank 0 is NOT the last stage

    config = PipelineStageConfig(
        rank=0,
        world_size=world_size,
        start_layer=0,
        end_layer=2,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    layers = nn.ModuleList([MockLayer(), MockLayer()])

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=embed_tokens,
        lm_head=None,
        final_norm=None,
    )

    # Input: token IDs [1, seq_len]
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    output, kv_cache = shard.forward(input_ids)

    assert output.shape == (1, seq_len, hidden_size), (
        f"First stage output shape mismatch: expected (1, {seq_len}, {hidden_size}), "
        f"got {output.shape}"
    )


@settings(max_examples=100)
@given(
    seq_len=seq_len_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
)
def test_middle_stage_accepts_and_outputs_hidden_state(
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Middle stage accepts [1, seq_len, hidden_size], outputs [1, seq_len, hidden_size].

    When rank is neither 0 nor world_size-1, the shard passes hidden_state through
    its local layers without embedding or lm_head.

    **Validates: Requirements 2.2, 2.4**
    """
    num_layers = 6
    world_size = 3  # 3 stages: rank 0 (first), rank 1 (middle), rank 2 (last)

    config = PipelineStageConfig(
        rank=1,
        world_size=world_size,
        start_layer=2,
        end_layer=4,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    layers = nn.ModuleList([MockLayer(), MockLayer()])

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=None,
        lm_head=None,
        final_norm=None,
    )

    # Input: hidden_state [1, seq_len, hidden_size]
    hidden_state = torch.randn(1, seq_len, hidden_size)
    output, kv_cache = shard.forward(hidden_state)

    assert output.shape == (1, seq_len, hidden_size), (
        f"Middle stage output shape mismatch: expected (1, {seq_len}, {hidden_size}), "
        f"got {output.shape}"
    )


@settings(max_examples=100)
@given(
    seq_len=seq_len_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
)
def test_last_stage_accepts_hidden_state_outputs_logits(
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Last stage accepts [1, seq_len, hidden_size], outputs [1, seq_len, vocab_size].

    When rank == world_size - 1 (and rank != 0), the shard runs through local layers,
    applies final_norm, and projects through lm_head to produce logits.

    **Validates: Requirements 2.3, 2.4**
    """
    num_layers = 4
    world_size = 2  # 2 stages: rank 0 (first), rank 1 (last)

    config = PipelineStageConfig(
        rank=1,
        world_size=world_size,
        start_layer=2,
        end_layer=4,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    layers = nn.ModuleList([MockLayer(), MockLayer()])
    final_norm = nn.LayerNorm(hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=None,
        lm_head=lm_head,
        final_norm=final_norm,
    )

    # Input: hidden_state [1, seq_len, hidden_size]
    hidden_state = torch.randn(1, seq_len, hidden_size)
    output, kv_cache = shard.forward(hidden_state)

    assert output.shape == (1, seq_len, vocab_size), (
        f"Last stage output shape mismatch: expected (1, {seq_len}, {vocab_size}), "
        f"got {output.shape}"
    )


@settings(max_examples=100)
@given(
    seq_len=seq_len_strategy,
    hidden_size=hidden_size_strategy,
    vocab_size=vocab_size_strategy,
)
def test_single_stage_accepts_token_ids_outputs_logits(
    seq_len: int,
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Single stage (world_size=1) accepts [1, seq_len] token IDs, outputs [1, seq_len, vocab_size].

    When world_size == 1, the shard is both first and last stage: it embeds token IDs,
    runs through layers, applies final_norm, and projects through lm_head.

    **Validates: Requirements 2.1, 2.3**
    """
    num_layers = 4

    config = PipelineStageConfig(
        rank=0,
        world_size=1,
        start_layer=0,
        end_layer=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        device="cpu",
    )

    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])
    final_norm = nn.LayerNorm(hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        final_norm=final_norm,
    )

    # Input: token IDs [1, seq_len]
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    output, kv_cache = shard.forward(input_ids)

    assert output.shape == (1, seq_len, vocab_size), (
        f"Single stage output shape mismatch: expected (1, {seq_len}, {vocab_size}), "
        f"got {output.shape}"
    )
