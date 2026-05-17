"""
Unit tests for initialize_pipeline_shard — the top-level orchestrator
that ties together manifest creation, tensor loading, module building,
and PipelineParallelShard construction.

Tests use mocks to avoid requiring real safetensors files or GPU hardware.

**Validates: Requirements 1.1, 1.3, 1.4, 1.9**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_PIPELINE_CONFIG_PATH = _ENGINE_DIR / "pipeline_config.py"
_LOCAL_SHARD_LOADER_PATH = _ENGINE_DIR / "local_shard_loader.py"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_config_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.pipeline_config", _PIPELINE_CONFIG_PATH
)
_loader_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.local_shard_loader",
    _LOCAL_SHARD_LOADER_PATH,
)

PipelineStageAssignment = _config_mod.PipelineStageAssignment
PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
LocalShardManifest = _loader_mod.LocalShardManifest
initialize_pipeline_shard = _loader_mod.initialize_pipeline_shard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(
    *,
    hidden_size: int = 256,
    vocab_size: int = 1000,
    num_hidden_layers: int = 8,
    num_attention_heads: int = 4,
    head_dim: int = 64,
) -> Any:
    """Create a minimal model config object with required attributes."""
    config = MagicMock()
    config.hidden_size = hidden_size
    config.vocab_size = vocab_size
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.head_dim = head_dim
    config.layer_types = None  # Will default to full_attention
    config.rope_theta = 10000.0
    config.max_position_embeddings = 4096
    config.partial_rotary_factor = 1.0
    return config


def _make_layer_distribution(
    layers_per_rank: tuple[int, ...] = (2, 2, 2, 2),
    total_layer_count: int = 8,
    rank_count: int = 4,
) -> Any:
    """Create a PipelineLayerDistribution."""
    return PipelineLayerDistribution(
        layers_per_rank=layers_per_rank,
        total_layer_count=total_layer_count,
        rank_count=rank_count,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitializePipelineShard:
    """Test the initialize_pipeline_shard orchestrator function."""

    def test_calls_steps_in_order(self, tmp_path: Path) -> None:
        """initialize_pipeline_shard calls manifest, load, validate, build in order."""
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config()
        device = torch.device("cpu")

        # Track call order
        call_order: list[str] = []

        # Create mock manifest
        mock_manifest = LocalShardManifest(
            tensor_names=["model.layers.0.self_attn.q_proj.weight"],
            tensor_to_file={"model.layers.0.self_attn.q_proj.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0, start_layer=0, end_layer=2,
                owns_embedding=True, owns_lm_head=False,
            ),
            total_tensor_count=10,
            local_tensor_count=1,
        )

        # Create mock tensors
        mock_tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
        }

        # Create mock modules
        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        def mock_load_manifest(**kwargs: Any) -> Any:
            call_order.append("load_manifest")
            return mock_manifest

        def mock_load_tensors(**kwargs: Any) -> Any:
            call_order.append("load_tensors")
            return mock_tensors

        def mock_validate(manifest: Any, tensors: Any) -> None:
            call_order.append("validate")

        def mock_build(**kwargs: Any) -> Any:
            call_order.append("build_modules")
            return mock_modules

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                side_effect=mock_load_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                side_effect=mock_load_tensors,
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
                side_effect=mock_validate,
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                side_effect=mock_build,
            ),
            patch(
                "torch.compile",
                side_effect=RuntimeError("skip compile in test"),
            ),
        ):
            shard = initialize_pipeline_shard(
                model_path=tmp_path,
                rank=0,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

        assert call_order == [
            "load_manifest",
            "load_tensors",
            "validate",
            "build_modules",
        ]
        # Verify the shard was created
        assert shard is not None
        assert shard.config.rank == 0
        assert shard.config.start_layer == 0
        assert shard.config.end_layer == 2

    def test_stage_assignment_derived_from_distribution(self, tmp_path: Path) -> None:
        """The stage assignment is correctly derived from the layer distribution."""
        # Rank 2 in a (2, 2, 2, 2) distribution gets layers [4, 6)
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config()
        device = torch.device("cpu")

        captured_assignment: list[Any] = []

        def mock_load_manifest(
            *, model_path: Any, stage_assignment: Any
        ) -> Any:
            captured_assignment.append(stage_assignment)
            return LocalShardManifest(
                tensor_names=["model.layers.4.self_attn.q_proj.weight"],
                tensor_to_file={"model.layers.4.self_attn.q_proj.weight": "shard.safetensors"},
                stage_assignment=stage_assignment,
                total_tensor_count=10,
                local_tensor_count=1,
            )

        mock_tensors = {
            "model.layers.4.self_attn.q_proj.weight": torch.randn(256, 256),
        }

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                side_effect=mock_load_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                return_value=mock_tensors,
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                return_value=mock_modules,
            ),
            patch(
                "torch.compile",
                side_effect=RuntimeError("skip compile in test"),
            ),
        ):
            shard = initialize_pipeline_shard(
                model_path=tmp_path,
                rank=2,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

        # Verify the assignment passed to load_manifest
        assert len(captured_assignment) == 1
        assignment = captured_assignment[0]
        assert assignment.rank == 2
        assert assignment.start_layer == 4
        assert assignment.end_layer == 6
        assert assignment.owns_embedding is False
        assert assignment.owns_lm_head is False

        # Verify the shard config
        assert shard.config.rank == 2
        assert shard.config.start_layer == 4
        assert shard.config.end_layer == 6

    def test_raises_on_missing_hidden_size(self, tmp_path: Path) -> None:
        """Raises ValueError when model_config.hidden_size is missing."""
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config(hidden_size=0)
        device = torch.device("cpu")

        mock_manifest = LocalShardManifest(
            tensor_names=["model.layers.0.self_attn.q_proj.weight"],
            tensor_to_file={"model.layers.0.self_attn.q_proj.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0, start_layer=0, end_layer=2,
                owns_embedding=True, owns_lm_head=False,
            ),
            total_tensor_count=10,
            local_tensor_count=1,
        )

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                return_value=mock_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                return_value={"model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256)},
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                return_value=mock_modules,
            ),
            pytest.raises(ValueError, match="hidden_size"),
        ):
            initialize_pipeline_shard(
                model_path=tmp_path,
                rank=0,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

    def test_raises_on_missing_vocab_size(self, tmp_path: Path) -> None:
        """Raises ValueError when model_config.vocab_size is missing."""
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config(vocab_size=0)
        device = torch.device("cpu")

        mock_manifest = LocalShardManifest(
            tensor_names=["model.layers.0.self_attn.q_proj.weight"],
            tensor_to_file={"model.layers.0.self_attn.q_proj.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0, start_layer=0, end_layer=2,
                owns_embedding=True, owns_lm_head=False,
            ),
            total_tensor_count=10,
            local_tensor_count=1,
        )

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                return_value=mock_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                return_value={"model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256)},
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                return_value=mock_modules,
            ),
            pytest.raises(ValueError, match="vocab_size"),
        ):
            initialize_pipeline_shard(
                model_path=tmp_path,
                rank=0,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

    def test_last_rank_owns_lm_head(self, tmp_path: Path) -> None:
        """The last rank (rank 3) gets owns_lm_head=True in its stage assignment."""
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config()
        device = torch.device("cpu")

        captured_assignment: list[Any] = []

        def mock_load_manifest(
            *, model_path: Any, stage_assignment: Any
        ) -> Any:
            captured_assignment.append(stage_assignment)
            return LocalShardManifest(
                tensor_names=["model.layers.6.self_attn.q_proj.weight"],
                tensor_to_file={"model.layers.6.self_attn.q_proj.weight": "shard.safetensors"},
                stage_assignment=stage_assignment,
                total_tensor_count=10,
                local_tensor_count=1,
            )

        mock_tensors = {
            "model.layers.6.self_attn.q_proj.weight": torch.randn(256, 256),
        }

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = torch.nn.Linear(256, 1000, bias=False)
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                side_effect=mock_load_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                return_value=mock_tensors,
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                return_value=mock_modules,
            ),
            patch(
                "torch.compile",
                side_effect=RuntimeError("skip compile in test"),
            ),
        ):
            shard = initialize_pipeline_shard(
                model_path=tmp_path,
                rank=3,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

        # Verify the assignment
        assert len(captured_assignment) == 1
        assignment = captured_assignment[0]
        assert assignment.rank == 3
        assert assignment.owns_lm_head is True
        assert assignment.owns_embedding is False

        # Verify the shard config
        assert shard.config.is_last_stage is True

    def test_passes_dtype_to_load_and_build(self, tmp_path: Path) -> None:
        """The dtype parameter is forwarded to load_tensors and build_modules."""
        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config()
        device = torch.device("cpu")

        captured_load_kwargs: list[dict[str, Any]] = []
        captured_build_kwargs: list[dict[str, Any]] = []

        mock_manifest = LocalShardManifest(
            tensor_names=["model.layers.0.self_attn.q_proj.weight"],
            tensor_to_file={"model.layers.0.self_attn.q_proj.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0, start_layer=0, end_layer=2,
                owns_embedding=True, owns_lm_head=False,
            ),
            total_tensor_count=10,
            local_tensor_count=1,
        )

        mock_tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
        }

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        def mock_load_tensors(**kwargs: Any) -> Any:
            captured_load_kwargs.append(kwargs)
            return mock_tensors

        def mock_build(**kwargs: Any) -> Any:
            captured_build_kwargs.append(kwargs)
            return mock_modules

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                return_value=mock_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                side_effect=mock_load_tensors,
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                side_effect=mock_build,
            ),
            patch(
                "torch.compile",
                side_effect=RuntimeError("skip compile in test"),
            ),
        ):
            initialize_pipeline_shard(
                model_path=tmp_path,
                rank=0,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
                dtype=torch.float32,
            )

        # Verify dtype was passed to load_tensors_from_manifest
        assert len(captured_load_kwargs) == 1
        assert captured_load_kwargs[0]["dtype"] == torch.float32

        # Verify dtype was passed to build_local_shard_modules
        assert len(captured_build_kwargs) == 1
        assert captured_build_kwargs[0]["dtype"] == torch.float32

    def test_returns_pipeline_parallel_shard(self, tmp_path: Path) -> None:
        """The return value is a PipelineParallelShard instance."""
        from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
            PipelineParallelShard,
        )

        layer_distribution = _make_layer_distribution()
        model_config = _make_model_config()
        device = torch.device("cpu")

        mock_manifest = LocalShardManifest(
            tensor_names=["model.layers.0.self_attn.q_proj.weight"],
            tensor_to_file={"model.layers.0.self_attn.q_proj.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0, start_layer=0, end_layer=2,
                owns_embedding=True, owns_lm_head=False,
            ),
            total_tensor_count=10,
            local_tensor_count=1,
        )

        mock_tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
        }

        mock_modules = MagicMock()
        mock_modules.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256)])
        mock_modules.embed_tokens = None
        mock_modules.lm_head = None
        mock_modules.final_norm = None
        mock_modules.rotary_emb = None

        with (
            patch.object(
                _loader_mod,
                "load_qwen_local_shard_from_safetensors",
                return_value=mock_manifest,
            ),
            patch.object(
                _loader_mod,
                "load_tensors_from_manifest",
                return_value=mock_tensors,
            ),
            patch.object(
                _loader_mod,
                "validate_loaded_tensors",
            ),
            patch.object(
                _loader_mod,
                "build_local_shard_modules",
                return_value=mock_modules,
            ),
            patch(
                "torch.compile",
                side_effect=RuntimeError("skip compile in test"),
            ),
        ):
            shard = initialize_pipeline_shard(
                model_path=tmp_path,
                rank=0,
                layer_distribution=layer_distribution,
                model_config=model_config,
                device=device,
            )

        assert isinstance(shard, PipelineParallelShard)
        assert shard.config.world_size == 4
        assert shard.config.hidden_size == 256
        assert shard.config.vocab_size == 1000
        assert shard.config.num_layers == 8
