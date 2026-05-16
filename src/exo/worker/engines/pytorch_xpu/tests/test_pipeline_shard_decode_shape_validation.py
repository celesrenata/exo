# Feature: pipeline-performance-optimization, Task: Decode activation shape validation
"""
Unit tests for decode activation shape stability and validation in PipelineParallelShard.

**Validates: Requirements 2.1, 2.2, 2.5**

Tests that the PipelineParallelShard correctly:
- Reports expected decode activation shape via decode_activation_shape property
- Tracks decode shape stability via is_decode_shape_stable property
- Validates output contiguity during decode steps
- Detects shape changes between steps
- Resets stability state on reset_state()
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_SHARD_PATH = _THIS_DIR.parent / "pipeline_parallel_shard.py"


def _load_pipeline_parallel_shard() -> types.ModuleType:
    """Load pipeline_parallel_shard.py directly from file, avoiding __init__.py."""
    module_name = "pipeline_parallel_shard_decode_shape_test"
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
    """Mock transformer layer that passes hidden_states through unchanged."""

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return (hidden_states, None)


# ---------------------------------------------------------------------------
# Helper to create a non-last-stage shard (middle stage)
# ---------------------------------------------------------------------------


def _make_middle_shard(hidden_size: int = 64) -> object:
    """Create a middle-stage shard (rank 1 of 3) for testing decode shape validation."""
    config = PipelineStageConfig(
        rank=1,
        world_size=3,
        start_layer=2,
        end_layer=4,
        hidden_size=hidden_size,
        vocab_size=100,
        num_layers=6,
        device="cpu",
    )
    layers = nn.ModuleList([MockLayer(), MockLayer()])
    return PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=None,
        lm_head=None,
        final_norm=None,
    )


# ---------------------------------------------------------------------------
# Tests: decode_activation_shape property
# ---------------------------------------------------------------------------


def test_decode_activation_shape_returns_expected_tuple() -> None:
    """decode_activation_shape returns (1, 1, hidden_size) for single-request mode."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)
    assert shard.decode_activation_shape == (1, 1, hidden_size)


def test_decode_activation_shape_varies_with_hidden_size() -> None:
    """decode_activation_shape reflects the configured hidden_size."""
    for hidden_size in [32, 128, 2560]:
        config = PipelineStageConfig(
            rank=0,
            world_size=2,
            start_layer=0,
            end_layer=2,
            hidden_size=hidden_size,
            vocab_size=100,
            num_layers=4,
            device="cpu",
        )
        layers = nn.ModuleList([MockLayer(), MockLayer()])
        embed = nn.Embedding(100, hidden_size)
        shard = PipelineParallelShard(
            layers=layers,
            config=config,
            embed_tokens=embed,
            lm_head=None,
            final_norm=None,
        )
        assert shard.decode_activation_shape == (1, 1, hidden_size)


# ---------------------------------------------------------------------------
# Tests: is_decode_shape_stable property
# ---------------------------------------------------------------------------


def test_is_decode_shape_stable_initially_false() -> None:
    """is_decode_shape_stable is False before any forward pass."""
    shard = _make_middle_shard()
    assert shard.is_decode_shape_stable is False


def test_is_decode_shape_stable_true_after_decode_step() -> None:
    """is_decode_shape_stable becomes True after a decode step with matching shape."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    # Simulate a decode step: seq_len=1, batch_size=1
    decode_input = torch.randn(1, 1, hidden_size)
    shard.forward(decode_input)

    assert shard.is_decode_shape_stable is True


def test_is_decode_shape_stable_false_during_prefill() -> None:
    """is_decode_shape_stable is False after a prefill step (seq_len > 1)."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    # Simulate a prefill step: seq_len=10
    prefill_input = torch.randn(1, 10, hidden_size)
    shard.forward(prefill_input)

    assert shard.is_decode_shape_stable is False


def test_is_decode_shape_stable_transitions_prefill_to_decode() -> None:
    """is_decode_shape_stable transitions from False (prefill) to True (decode)."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    # Prefill
    prefill_input = torch.randn(1, 10, hidden_size)
    shard.forward(prefill_input)
    assert shard.is_decode_shape_stable is False

    # Decode
    decode_input = torch.randn(1, 1, hidden_size)
    shard.forward(decode_input)
    assert shard.is_decode_shape_stable is True


def test_is_decode_shape_stable_resets_on_state_reset() -> None:
    """is_decode_shape_stable resets to False after reset_state()."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    # Enter steady state
    decode_input = torch.randn(1, 1, hidden_size)
    shard.forward(decode_input)
    assert shard.is_decode_shape_stable is True

    # Reset
    shard.reset_state()
    assert shard.is_decode_shape_stable is False


# ---------------------------------------------------------------------------
# Tests: output contiguity
# ---------------------------------------------------------------------------


def test_decode_output_is_contiguous() -> None:
    """Decode output from a non-last-stage shard is always contiguous."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    decode_input = torch.randn(1, 1, hidden_size)
    output, _ = shard.forward(decode_input)

    assert output.is_contiguous()


def test_decode_output_contiguous_with_non_contiguous_layer_output() -> None:
    """Even if a layer produces non-contiguous output, the shard makes it contiguous."""
    hidden_size = 64

    class NonContiguousLayer(nn.Module):
        """Layer that returns a non-contiguous tensor (transposed view)."""

        def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
            # Create a non-contiguous tensor by transposing and transposing back
            # via a view that breaks contiguity
            batch, seq, hidden = hidden_states.shape
            # Expand then slice to create non-contiguous memory
            expanded = hidden_states.unsqueeze(2).expand(batch, seq, 2, hidden)
            non_contig = expanded[:, :, 0, :]  # This slice is non-contiguous
            return (non_contig, None)

    config = PipelineStageConfig(
        rank=1,
        world_size=3,
        start_layer=2,
        end_layer=3,
        hidden_size=hidden_size,
        vocab_size=100,
        num_layers=6,
        device="cpu",
    )
    layers = nn.ModuleList([NonContiguousLayer()])
    shard = PipelineParallelShard(
        layers=layers,
        config=config,
        embed_tokens=None,
        lm_head=None,
        final_norm=None,
    )

    decode_input = torch.randn(1, 1, hidden_size)
    output, _ = shard.forward(decode_input)

    assert output.is_contiguous(), "Decode output must be contiguous for fast-path communication"


# ---------------------------------------------------------------------------
# Tests: last stage does NOT get decode validation (returns logits)
# ---------------------------------------------------------------------------


def test_last_stage_skips_decode_validation() -> None:
    """Last stage (returns logits) does not apply decode shape validation."""
    hidden_size = 64
    vocab_size = 100

    config = PipelineStageConfig(
        rank=2,
        world_size=3,
        start_layer=4,
        end_layer=6,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=6,
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

    # Decode step on last stage
    decode_input = torch.randn(1, 1, hidden_size)
    output, _ = shard.forward(decode_input)

    # Last stage outputs logits, not hidden state — shape is (1, 1, vocab_size)
    assert output.shape == (1, 1, vocab_size)
    # is_decode_shape_stable should remain False since validation is skipped
    assert shard.is_decode_shape_stable is False


# ---------------------------------------------------------------------------
# Tests: shape change detection
# ---------------------------------------------------------------------------


def test_shape_change_exits_steady_state() -> None:
    """A shape change between decode steps exits steady state."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    # Enter steady state with batch_size=1
    decode_input = torch.randn(1, 1, hidden_size)
    shard.forward(decode_input)
    assert shard.is_decode_shape_stable is True

    # New prefill with different shape exits steady state
    prefill_input = torch.randn(1, 5, hidden_size)
    shard.forward(prefill_input)
    assert shard.is_decode_shape_stable is False


def test_multiple_decode_steps_remain_stable() -> None:
    """Multiple consecutive decode steps with same shape remain in steady state."""
    hidden_size = 64
    shard = _make_middle_shard(hidden_size=hidden_size)

    for _ in range(5):
        decode_input = torch.randn(1, 1, hidden_size)
        shard.forward(decode_input)

    assert shard.is_decode_shape_stable is True
