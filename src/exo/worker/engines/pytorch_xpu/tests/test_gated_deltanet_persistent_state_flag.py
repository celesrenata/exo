"""
Unit tests for the enable_gated_deltanet_persistent_state compatibility flag.

Verifies that PipelineParallelShard correctly gates optimization features
based on the PytorchXpuOptimizationConfiguration flag:
- When enabled (default): GatedDeltaNetCache, DecodeOutputBufferPool, and
  persistent state initialization are active.
- When disabled: plain DynamicCache is used, no buffer pool, no persistent
  state initialization.

**Validates: Requirements 5.1, 5.2, 5.7, 5.8**
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

_config_mod = _load_module("pipeline_config_flag_test", _PIPELINE_CONFIG_PATH)
_shard_mod = _load_module("pipeline_shard_flag_test", _PIPELINE_SHARD_PATH)

PytorchXpuOptimizationConfiguration = _config_mod.PytorchXpuOptimizationConfiguration
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


def _make_shard(
    optimization_config: PytorchXpuOptimizationConfiguration | None = None,
    num_layers: int = 2,
) -> PipelineParallelShard:
    """Create a PipelineParallelShard with the given optimization config."""
    config = PipelineStageConfig(
        rank=1,
        world_size=4,
        start_layer=16,
        end_layer=16 + num_layers,
        hidden_size=HIDDEN_SIZE,
        vocab_size=100,
        num_layers=64,
        device="cpu",
    )
    layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])

    with patch("torch.compile", side_effect=RuntimeError("skip compile in test")):
        shard = PipelineParallelShard(
            layers=layers,
            config=config,
            embed_tokens=None,
            lm_head=None,
            final_norm=None,
            optimization_config=optimization_config,
        )
    return shard


# ===========================================================================
# Tests
# ===========================================================================


class TestGatedDeltaNetPersistentStateFlag:
    """Test the enable_gated_deltanet_persistent_state compatibility flag."""

    def test_flag_defaults_to_true_when_no_config(self) -> None:
        """When no optimization_config is provided, the flag defaults to True."""
        shard = _make_shard(optimization_config=None)
        assert shard.gated_deltanet_persistent_state_enabled is True

    def test_flag_enabled_creates_gated_deltanet_cache(self) -> None:
        """When flag is enabled, GatedDeltaNetCache wrapper is created."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=True,
        )
        shard = _make_shard(optimization_config=config)
        assert shard.gated_deltanet_persistent_state_enabled is True
        # GatedDeltaNetCache should be created (if transformers is available)
        # The cache wrapper is the optimization path
        assert shard.gated_deltanet_cache is not None

    def test_flag_disabled_no_gated_deltanet_cache(self) -> None:
        """When flag is disabled, no GatedDeltaNetCache wrapper is created."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=False,
        )
        shard = _make_shard(optimization_config=config)
        assert shard.gated_deltanet_persistent_state_enabled is False
        # GatedDeltaNetCache should NOT be created
        assert shard.gated_deltanet_cache is None

    def test_flag_disabled_no_buffer_pool(self) -> None:
        """When flag is disabled, DecodeOutputBufferPool is not created."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=False,
        )
        shard = _make_shard(optimization_config=config)
        assert shard.decode_output_buffer_pool is None

    def test_flag_enabled_creates_buffer_pool(self) -> None:
        """When flag is enabled, DecodeOutputBufferPool is created."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=True,
        )
        shard = _make_shard(optimization_config=config)
        # Buffer pool should be created when optimization is enabled
        assert shard.decode_output_buffer_pool is not None

    def test_flag_disabled_still_has_hf_cache(self) -> None:
        """When flag is disabled, plain DynamicCache is still available for layers."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=False,
        )
        shard = _make_shard(optimization_config=config)
        # _hf_cache should still exist (plain DynamicCache) for layer state
        assert shard._hf_cache is not None

    def test_flag_disabled_initialize_request_skips_persistent_state(self) -> None:
        """When flag is disabled, initialize_request sets ID but skips state init."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=False,
        )
        shard = _make_shard(optimization_config=config)
        shard.initialize_request("test-request-123")
        # Request ID should be set
        assert shard.current_request_id == "test-request-123"
        # No GatedDeltaNetCache means no persistent states
        assert shard.gated_deltanet_cache is None

    def test_flag_enabled_initialize_request_sets_id(self) -> None:
        """When flag is enabled, initialize_request sets the request ID."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=True,
        )
        shard = _make_shard(optimization_config=config)
        shard.initialize_request("test-request-456")
        assert shard.current_request_id == "test-request-456"

    def test_flag_disabled_reset_state_works(self) -> None:
        """When flag is disabled, reset_state clears state without errors."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=False,
        )
        shard = _make_shard(optimization_config=config)
        shard.initialize_request("test-request")
        # Should not raise
        shard.reset_state()
        assert shard.current_request_id is None

    def test_flag_enabled_reset_state_works(self) -> None:
        """When flag is enabled, reset_state clears all optimization state."""
        config = PytorchXpuOptimizationConfiguration(
            enable_gated_deltanet_persistent_state=True,
        )
        shard = _make_shard(optimization_config=config)
        shard.initialize_request("test-request")
        shard.reset_state()
        assert shard.current_request_id is None
