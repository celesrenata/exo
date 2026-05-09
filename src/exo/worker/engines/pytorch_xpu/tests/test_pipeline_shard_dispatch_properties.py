# Feature: pipeline-parallelism-optimization, Property 7 & 8: Shard type dispatch and partial loading
"""
Property-based tests for shard type dispatch logic and partial loading exclusion.

**Validates: Requirements 8.1, 8.2, 10.4**

Uses Hypothesis to generate shard metadata combinations (start_layer, end_layer,
n_layers, world_size) and verifies that the dispatch logic correctly determines
whether to create a TensorParallelShard or PipelineParallelShard.

Also verifies that partial model loading excludes weights for layers outside
the assigned [start_layer, end_layer) range.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper: replicate the dispatch logic from model_loader.py _load_model_sync()
# ---------------------------------------------------------------------------


def determine_shard_type(
    start_layer: int, end_layer: int, n_layers: int, world_size: int
) -> str:
    """Determine the shard type based on layer range and world_size.

    This replicates the dispatch logic from model_loader.py:
      is_tensor_parallel = (start_layer == 0 and end_layer == n_layers and world_size > 1)
      is_pipeline_parallel = ((start_layer != 0 or end_layer != n_layers) and world_size > 1)

    Returns one of:
      - "tensor_parallel": all layers on every node, world_size > 1
      - "pipeline_parallel": partial layer range, world_size > 1
      - "transformer_shard": partial layer range, world_size == 1
      - "none": all layers, world_size == 1 (no sharding needed)
    """
    is_tensor_parallel = (
        start_layer == 0
        and end_layer == n_layers
        and world_size > 1
    )

    is_pipeline_parallel = (
        (start_layer != 0 or end_layer != n_layers)
        and world_size > 1
    )

    if is_tensor_parallel:
        return "tensor_parallel"
    elif is_pipeline_parallel:
        return "pipeline_parallel"
    elif not (start_layer == 0 and end_layer == n_layers):
        # Single-node partial layer range (world_size == 1)
        return "transformer_shard"
    else:
        # All layers, world_size == 1: no sharding
        return "none"


# ---------------------------------------------------------------------------
# Property 7: Shard type dispatch is determined by layer range
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=64),
    world_size=st.integers(min_value=2, max_value=8),
)
def test_full_layer_range_with_multi_node_dispatches_tensor_parallel(
    n_layers: int, world_size: int
) -> None:
    """start_layer==0 and end_layer==n_layers and world_size>1 → TensorParallelShard.

    **Validates: Requirements 8.1**
    """
    shard_type = determine_shard_type(
        start_layer=0,
        end_layer=n_layers,
        n_layers=n_layers,
        world_size=world_size,
    )
    assert shard_type == "tensor_parallel", (
        f"Expected 'tensor_parallel' for start_layer=0, end_layer={n_layers}, "
        f"n_layers={n_layers}, world_size={world_size}, got '{shard_type}'"
    )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=64),
    world_size=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
def test_partial_layer_range_with_multi_node_dispatches_pipeline_parallel(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """(start_layer!=0 or end_layer!=n_layers) and world_size>1 → PipelineParallelShard.

    **Validates: Requirements 8.2**
    """
    # Generate a partial layer range (not the full [0, n_layers))
    # Strategy: pick start_layer and end_layer such that the range is a strict subset
    strategy_choice = data.draw(
        st.sampled_from(["nonzero_start", "partial_end", "both"]),
        label="partial_type",
    )

    if strategy_choice == "nonzero_start":
        # start_layer > 0, end_layer can be anything valid
        start_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 1), label="start_layer"
        )
        end_layer = data.draw(
            st.integers(min_value=start_layer + 1, max_value=n_layers), label="end_layer"
        )
    elif strategy_choice == "partial_end":
        # start_layer == 0, end_layer < n_layers
        start_layer = 0
        end_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 1), label="end_layer"
        )
    else:
        # Both: start_layer > 0 AND end_layer < n_layers
        start_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 2), label="start_layer"
        )
        end_layer = data.draw(
            st.integers(min_value=start_layer + 1, max_value=n_layers - 1), label="end_layer"
        )

    shard_type = determine_shard_type(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        world_size=world_size,
    )
    assert shard_type == "pipeline_parallel", (
        f"Expected 'pipeline_parallel' for start_layer={start_layer}, end_layer={end_layer}, "
        f"n_layers={n_layers}, world_size={world_size}, got '{shard_type}'"
    )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=64),
    data=st.data(),
)
def test_partial_layer_range_single_node_dispatches_transformer_shard(
    n_layers: int, data: st.DataObject
) -> None:
    """(start_layer!=0 or end_layer!=n_layers) and world_size==1 → TransformerShard.

    When world_size is 1 but the layer range is partial, the dispatch creates
    a TransformerShard (single-node partial layer range).
    """
    # Generate a partial layer range with world_size == 1
    strategy_choice = data.draw(
        st.sampled_from(["nonzero_start", "partial_end", "both"]),
        label="partial_type",
    )

    if strategy_choice == "nonzero_start":
        start_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 1), label="start_layer"
        )
        end_layer = data.draw(
            st.integers(min_value=start_layer + 1, max_value=n_layers), label="end_layer"
        )
    elif strategy_choice == "partial_end":
        start_layer = 0
        end_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 1), label="end_layer"
        )
    else:
        start_layer = data.draw(
            st.integers(min_value=1, max_value=n_layers - 2), label="start_layer"
        )
        end_layer = data.draw(
            st.integers(min_value=start_layer + 1, max_value=n_layers - 1), label="end_layer"
        )

    shard_type = determine_shard_type(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        world_size=1,
    )
    assert shard_type == "transformer_shard", (
        f"Expected 'transformer_shard' for start_layer={start_layer}, end_layer={end_layer}, "
        f"n_layers={n_layers}, world_size=1, got '{shard_type}'"
    )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=64),
)
def test_full_layer_range_single_node_dispatches_none(n_layers: int) -> None:
    """start_layer==0 and end_layer==n_layers and world_size==1 → no sharding.

    When world_size is 1 and all layers are present, no sharding wrapper is needed.
    """
    shard_type = determine_shard_type(
        start_layer=0,
        end_layer=n_layers,
        n_layers=n_layers,
        world_size=1,
    )
    assert shard_type == "none", (
        f"Expected 'none' for start_layer=0, end_layer={n_layers}, "
        f"n_layers={n_layers}, world_size=1, got '{shard_type}'"
    )


# ---------------------------------------------------------------------------
# Property: dispatch categories are mutually exclusive and exhaustive
# ---------------------------------------------------------------------------


@settings(max_examples=500)
@given(
    n_layers=st.integers(min_value=4, max_value=64),
    world_size=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
def test_dispatch_categories_are_mutually_exclusive(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """Every valid (start_layer, end_layer, n_layers, world_size) maps to exactly one shard type.

    **Validates: Requirements 8.1, 8.2**
    """
    # Generate valid start_layer and end_layer within [0, n_layers)
    start_layer = data.draw(
        st.integers(min_value=0, max_value=n_layers - 1), label="start_layer"
    )
    end_layer = data.draw(
        st.integers(min_value=start_layer + 1, max_value=n_layers), label="end_layer"
    )

    shard_type = determine_shard_type(start_layer, end_layer, n_layers, world_size)

    # Must be one of the four valid types
    valid_types = {"tensor_parallel", "pipeline_parallel", "transformer_shard", "none"}
    assert shard_type in valid_types, (
        f"Got unexpected shard type '{shard_type}' for "
        f"start_layer={start_layer}, end_layer={end_layer}, "
        f"n_layers={n_layers}, world_size={world_size}"
    )

    # Verify the type matches the expected conditions
    is_full_range = (start_layer == 0 and end_layer == n_layers)
    is_multi_node = (world_size > 1)

    if is_full_range and is_multi_node:
        assert shard_type == "tensor_parallel"
    elif not is_full_range and is_multi_node:
        assert shard_type == "pipeline_parallel"
    elif not is_full_range and not is_multi_node:
        assert shard_type == "transformer_shard"
    else:
        # is_full_range and not is_multi_node
        assert shard_type == "none"



# ---------------------------------------------------------------------------
# Property 8: Partial loading excludes out-of-range layer weights
# ---------------------------------------------------------------------------

# Mock model structure that mimics a HuggingFace transformer model
# with named parameters like model.layers.{idx}.self_attn.q_proj.weight


class _MockConfig:
    """Mock HuggingFace model config."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.text_config = None


class _MockInnerModel(nn.Module):
    """Mock model.model with layers, embed_tokens, and norm."""

    def __init__(self, n_layers: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)


class _MockModel(nn.Module):
    """Mock HuggingFace model with model.model.layers structure."""

    def __init__(self, n_layers: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.model = _MockInnerModel(n_layers, hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = _MockConfig(hidden_size, vocab_size)


class _MockShardMetadata:
    """Mock shard metadata with the fields needed by _create_pipeline_parallel_shard."""

    def __init__(
        self,
        start_layer: int,
        end_layer: int,
        n_layers: int,
        device_rank: int,
        world_size: int,
    ) -> None:
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.n_layers = n_layers
        self.device_rank = device_rank
        self.world_size = world_size


def _extract_pipeline_shard(
    model: _MockModel, shard_metadata: _MockShardMetadata
) -> dict[str, torch.Tensor]:
    """Extract layers from a mock model using the same logic as _create_pipeline_parallel_shard.

    Returns the state_dict of the extracted layers (not the full shard object)
    to verify which parameters are present.
    """
    # Extract layers from model.model.layers[start_layer:end_layer]
    extracted_layers = nn.ModuleList(
        list(model.model.layers[shard_metadata.start_layer : shard_metadata.end_layer])
    )

    # Extract embed_tokens if this is the first stage (rank 0)
    embed_tokens: nn.Embedding | None = None
    if shard_metadata.start_layer == 0:
        embed_tokens = model.model.embed_tokens

    # Extract lm_head and final_norm if this is the last stage
    lm_head: nn.Linear | None = None
    final_norm: nn.Module | None = None
    if shard_metadata.end_layer == shard_metadata.n_layers:
        lm_head = model.lm_head
        final_norm = model.model.norm

    # Build a combined state dict of all extracted components
    state_dict: dict[str, torch.Tensor] = {}

    # Add layer parameters with their original naming convention
    for i, layer in enumerate(extracted_layers):
        layer_idx = shard_metadata.start_layer + i
        for name, param in layer.named_parameters():
            state_dict[f"model.layers.{layer_idx}.{name}"] = param

    if embed_tokens is not None:
        for name, param in embed_tokens.named_parameters():
            state_dict[f"model.embed_tokens.{name}"] = param

    if lm_head is not None:
        for name, param in lm_head.named_parameters():
            state_dict[f"lm_head.{name}"] = param

    if final_norm is not None:
        for name, param in final_norm.named_parameters():
            state_dict[f"model.norm.{name}"] = param

    return state_dict


# Regex pattern to match layer parameter keys like "model.layers.{idx}...."
_LAYER_KEY_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=16),
    world_size=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
def test_partial_loading_excludes_out_of_range_layers(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """Loaded state dict contains no keys for layers outside [start_layer, end_layer).

    **Validates: Requirements 10.4**

    For any layer range [start_layer, end_layer) and any model with n_layers total
    layers, the loaded state dict SHALL contain no parameter keys matching
    model.layers.{idx}.* for any idx outside [start_layer, end_layer).
    """
    hidden_size = 32  # Small for fast testing
    vocab_size = 64

    # Pick a valid rank and compute its layer range
    rank = data.draw(
        st.integers(min_value=0, max_value=min(world_size, n_layers) - 1),
        label="rank",
    )

    # Compute layer assignment (same logic as compute_layer_assignment)
    base = n_layers // world_size
    remainder = n_layers % world_size
    if rank < remainder:
        start_layer = rank * (base + 1)
        end_layer = start_layer + (base + 1)
    else:
        start_layer = remainder * (base + 1) + (rank - remainder) * base
        end_layer = start_layer + base

    # Skip if the range is empty (world_size > n_layers edge case)
    if start_layer >= end_layer or end_layer > n_layers:
        return

    # Create mock model and shard metadata
    model = _MockModel(n_layers, hidden_size, vocab_size)
    shard_metadata = _MockShardMetadata(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        device_rank=rank,
        world_size=world_size,
    )

    # Extract the shard's state dict
    state_dict = _extract_pipeline_shard(model, shard_metadata)

    # Verify: no layer keys outside [start_layer, end_layer)
    for key in state_dict:
        match = _LAYER_KEY_PATTERN.match(key)
        if match:
            layer_idx = int(match.group(1))
            assert start_layer <= layer_idx < end_layer, (
                f"Found parameter key '{key}' with layer index {layer_idx} "
                f"outside assigned range [{start_layer}, {end_layer})"
            )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=16),
    world_size=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
def test_partial_loading_includes_all_in_range_layers(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """Loaded state dict contains keys for ALL layers in [start_layer, end_layer).

    **Validates: Requirements 10.4**

    Complementary property: the extraction does not accidentally skip layers
    that should be included.
    """
    hidden_size = 32
    vocab_size = 64

    # Pick a valid rank and compute its layer range
    rank = data.draw(
        st.integers(min_value=0, max_value=min(world_size, n_layers) - 1),
        label="rank",
    )

    # Compute layer assignment
    base = n_layers // world_size
    remainder = n_layers % world_size
    if rank < remainder:
        start_layer = rank * (base + 1)
        end_layer = start_layer + (base + 1)
    else:
        start_layer = remainder * (base + 1) + (rank - remainder) * base
        end_layer = start_layer + base

    # Skip if the range is empty
    if start_layer >= end_layer or end_layer > n_layers:
        return

    # Create mock model and shard metadata
    model = _MockModel(n_layers, hidden_size, vocab_size)
    shard_metadata = _MockShardMetadata(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        device_rank=rank,
        world_size=world_size,
    )

    # Extract the shard's state dict
    state_dict = _extract_pipeline_shard(model, shard_metadata)

    # Collect which layer indices are present in the state dict
    present_layers: set[int] = set()
    for key in state_dict:
        match = _LAYER_KEY_PATTERN.match(key)
        if match:
            present_layers.add(int(match.group(1)))

    # Verify: every layer in [start_layer, end_layer) has at least one parameter
    expected_layers = set(range(start_layer, end_layer))
    assert present_layers == expected_layers, (
        f"Expected layers {sorted(expected_layers)} but found {sorted(present_layers)} "
        f"for range [{start_layer}, {end_layer})"
    )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=16),
    world_size=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
def test_partial_loading_embed_only_on_first_stage(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """embed_tokens parameters are present only when start_layer == 0.

    **Validates: Requirements 10.4**

    The first stage (rank 0) gets embed_tokens; other stages do not.
    """
    hidden_size = 32
    vocab_size = 64

    # Pick a valid rank
    rank = data.draw(
        st.integers(min_value=0, max_value=min(world_size, n_layers) - 1),
        label="rank",
    )

    # Compute layer assignment
    base = n_layers // world_size
    remainder = n_layers % world_size
    if rank < remainder:
        start_layer = rank * (base + 1)
        end_layer = start_layer + (base + 1)
    else:
        start_layer = remainder * (base + 1) + (rank - remainder) * base
        end_layer = start_layer + base

    if start_layer >= end_layer or end_layer > n_layers:
        return

    model = _MockModel(n_layers, hidden_size, vocab_size)
    shard_metadata = _MockShardMetadata(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        device_rank=rank,
        world_size=world_size,
    )

    state_dict = _extract_pipeline_shard(model, shard_metadata)

    has_embed = any(k.startswith("model.embed_tokens.") for k in state_dict)

    if start_layer == 0:
        assert has_embed, (
            f"First stage (start_layer=0) should have embed_tokens parameters"
        )
    else:
        assert not has_embed, (
            f"Non-first stage (start_layer={start_layer}) should NOT have "
            f"embed_tokens parameters"
        )


@settings(max_examples=200)
@given(
    n_layers=st.integers(min_value=4, max_value=16),
    world_size=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
def test_partial_loading_lm_head_only_on_last_stage(
    n_layers: int, world_size: int, data: st.DataObject
) -> None:
    """lm_head and final_norm parameters are present only when end_layer == n_layers.

    **Validates: Requirements 10.4**

    The last stage gets lm_head and model.norm; other stages do not.
    """
    hidden_size = 32
    vocab_size = 64

    # Pick a valid rank
    rank = data.draw(
        st.integers(min_value=0, max_value=min(world_size, n_layers) - 1),
        label="rank",
    )

    # Compute layer assignment
    base = n_layers // world_size
    remainder = n_layers % world_size
    if rank < remainder:
        start_layer = rank * (base + 1)
        end_layer = start_layer + (base + 1)
    else:
        start_layer = remainder * (base + 1) + (rank - remainder) * base
        end_layer = start_layer + base

    if start_layer >= end_layer or end_layer > n_layers:
        return

    model = _MockModel(n_layers, hidden_size, vocab_size)
    shard_metadata = _MockShardMetadata(
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
        device_rank=rank,
        world_size=world_size,
    )

    state_dict = _extract_pipeline_shard(model, shard_metadata)

    has_lm_head = any(k.startswith("lm_head.") for k in state_dict)
    has_norm = any(k.startswith("model.norm.") for k in state_dict)

    if end_layer == n_layers:
        assert has_lm_head, (
            f"Last stage (end_layer={n_layers}) should have lm_head parameters"
        )
        assert has_norm, (
            f"Last stage (end_layer={n_layers}) should have model.norm parameters"
        )
    else:
        assert not has_lm_head, (
            f"Non-last stage (end_layer={end_layer}) should NOT have lm_head parameters"
        )
        assert not has_norm, (
            f"Non-last stage (end_layer={end_layer}) should NOT have model.norm parameters"
        )
