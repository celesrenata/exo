"""
Unit tests for PipelineParallelShard stage assignment integration.

Verifies:
- The layer count assertion rejects mismatched layer counts.
- The `loaded_layer_range` property exposes the correct (start, end) tuple.
- The `stage_assignment` property returns a valid PipelineStageAssignment.
- The `from_local_shard_modules` classmethod creates a shard from LocalShardModules.

**Validates: Requirements 1.1, 1.4, 2.1**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_PIPELINE_CONFIG_PATH = _PARENT_DIR / "pipeline_config.py"
_PIPELINE_SHARD_PATH = _PARENT_DIR / "pipeline_parallel_shard.py"


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# These tests require torch
torch = pytest.importorskip("torch")
nn = torch.nn

_config_mod = _load_module("pipeline_config_stage_test", _PIPELINE_CONFIG_PATH)
_shard_mod = _load_module("pipeline_shard_stage_test", _PIPELINE_SHARD_PATH)

PipelineStageAssignment = _config_mod.PipelineStageAssignment
PipelineParallelShard = _shard_mod.PipelineParallelShard
PipelineStageConfig = _shard_mod.PipelineStageConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64


class MockLayer(nn.Module):
    """Minimal layer that passes hidden states through unchanged."""

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return hidden_states, None


def _make_config(
    rank: int = 1,
    world_size: int = 4,
    start_layer: int = 16,
    end_layer: int = 32,
) -> PipelineStageConfig:
    """Create a PipelineStageConfig for testing."""
    return PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=HIDDEN_SIZE,
        vocab_size=100,
        num_layers=64,
        device="cpu",
    )


def _make_shard(
    num_layers: int = 16,
    rank: int = 1,
    world_size: int = 4,
    start_layer: int = 16,
    end_layer: int = 32,
) -> PipelineParallelShard:
    """Create a PipelineParallelShard with matching layer count."""
    config = _make_config(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
    )
    layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    with patch("torch.compile", side_effect=RuntimeError("skip compile in test")):
        shard = PipelineParallelShard(
            layers=layers,
            config=config,
            embed_tokens=None,
            lm_head=None,
            final_norm=None,
        )
    return shard


# ===========================================================================
# Tests
# ===========================================================================


class TestLayerCountAssertion:
    """Test that __init__ asserts len(layers) == config.num_local_layers."""

    def test_matching_layer_count_succeeds(self) -> None:
        """Creating a shard with correct layer count does not raise."""
        shard = _make_shard(num_layers=16, start_layer=16, end_layer=32)
        assert len(shard.layers) == 16

    def test_too_few_layers_raises(self) -> None:
        """Creating a shard with fewer layers than expected raises AssertionError."""
        config = _make_config(start_layer=16, end_layer=32)  # expects 16 layers
        layers = nn.ModuleList([MockLayer() for _ in range(10)])  # only 10

        with pytest.raises(AssertionError, match="Expected 16 layers for rank 1, got 10"):
            with patch("torch.compile", side_effect=RuntimeError("skip")):
                PipelineParallelShard(
                    layers=layers,
                    config=config,
                    embed_tokens=None,
                    lm_head=None,
                    final_norm=None,
                )

    def test_too_many_layers_raises(self) -> None:
        """Creating a shard with more layers than expected raises AssertionError."""
        config = _make_config(start_layer=0, end_layer=4)  # expects 4 layers
        layers = nn.ModuleList([MockLayer() for _ in range(8)])  # 8 layers

        with pytest.raises(AssertionError, match="Expected 4 layers for rank 1, got 8"):
            with patch("torch.compile", side_effect=RuntimeError("skip")):
                PipelineParallelShard(
                    layers=layers,
                    config=config,
                    embed_tokens=None,
                    lm_head=None,
                    final_norm=None,
                )


class TestLoadedLayerRange:
    """Test the loaded_layer_range property."""

    def test_returns_start_end_tuple(self) -> None:
        """loaded_layer_range returns (start_layer, end_layer) from config."""
        shard = _make_shard(num_layers=16, start_layer=16, end_layer=32)
        assert shard.loaded_layer_range == (16, 32)

    def test_first_rank_range(self) -> None:
        """First rank has range starting at 0."""
        shard = _make_shard(
            num_layers=16, rank=0, start_layer=0, end_layer=16
        )
        assert shard.loaded_layer_range == (0, 16)

    def test_last_rank_range(self) -> None:
        """Last rank has range ending at total layers."""
        shard = _make_shard(
            num_layers=16, rank=3, start_layer=48, end_layer=64
        )
        assert shard.loaded_layer_range == (48, 64)


class TestStageAssignment:
    """Test the stage_assignment property."""

    def test_returns_pipeline_stage_assignment(self) -> None:
        """stage_assignment returns a PipelineStageAssignment instance."""
        shard = _make_shard(num_layers=16, rank=1, start_layer=16, end_layer=32)
        assignment = shard.stage_assignment
        assert isinstance(assignment, PipelineStageAssignment)

    def test_assignment_matches_config(self) -> None:
        """stage_assignment fields match the PipelineStageConfig values."""
        shard = _make_shard(num_layers=16, rank=2, start_layer=32, end_layer=48)
        assignment = shard.stage_assignment
        assert assignment.rank == 2
        assert assignment.start_layer == 32
        assert assignment.end_layer == 48
        assert assignment.owns_embedding is False
        assert assignment.owns_lm_head is False

    def test_first_rank_owns_embedding(self) -> None:
        """First rank's stage_assignment has owns_embedding=True."""
        shard = _make_shard(
            num_layers=16, rank=0, start_layer=0, end_layer=16
        )
        assignment = shard.stage_assignment
        assert assignment.owns_embedding is True
        assert assignment.owns_lm_head is False

    def test_last_rank_owns_lm_head(self) -> None:
        """Last rank's stage_assignment has owns_lm_head=True."""
        shard = _make_shard(
            num_layers=16, rank=3, start_layer=48, end_layer=64
        )
        assignment = shard.stage_assignment
        assert assignment.owns_embedding is False
        assert assignment.owns_lm_head is True

    def test_num_local_layers_matches(self) -> None:
        """stage_assignment.num_local_layers matches the shard's layer count."""
        shard = _make_shard(num_layers=16, rank=1, start_layer=16, end_layer=32)
        assert shard.stage_assignment.num_local_layers == 16


class TestFromLocalShardModules:
    """Test the from_local_shard_modules classmethod."""

    def test_creates_shard_from_modules(self) -> None:
        """from_local_shard_modules creates a working PipelineParallelShard."""
        # Import LocalShardModules
        _loader_path = _PARENT_DIR / "local_shard_loader.py"
        _loader_mod = _load_module("local_shard_loader_test", _loader_path)
        LocalShardModules = _loader_mod.LocalShardModules

        num_layers = 4
        layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])
        config = _make_config(rank=0, start_layer=0, end_layer=4)

        modules = LocalShardModules(
            layers=layers,
            embed_tokens=nn.Embedding(100, HIDDEN_SIZE),
            lm_head=None,
            final_norm=None,
            rotary_emb=None,
            layer_types=["full_attention"] * num_layers,
            global_layer_indices=list(range(num_layers)),
        )

        with patch("torch.compile", side_effect=RuntimeError("skip compile in test")):
            shard = PipelineParallelShard.from_local_shard_modules(
                modules=modules,
                config=config,
            )

        assert len(shard.layers) == num_layers
        assert shard.embed_tokens is not None
        assert shard.lm_head is None
        assert shard.loaded_layer_range == (0, 4)
        assert shard.stage_assignment.rank == 0
        assert shard.stage_assignment.owns_embedding is True
