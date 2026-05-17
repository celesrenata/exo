"""
Unit tests for validate_pipeline_distribution startup validation.

Tests that configuration errors are caught at startup rather than
producing silent failures during inference.

**Validates: Requirements 1.10**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

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


# Load pipeline_config under its canonical import path
_config_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.pipeline_config", _PIPELINE_CONFIG_PATH
)
_loader_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.local_shard_loader",
    _LOCAL_SHARD_LOADER_PATH,
)

PipelineStageAssignment = _config_mod.PipelineStageAssignment
PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
validate_pipeline_distribution = _loader_mod.validate_pipeline_distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_weight_map(num_layers: int = 8) -> dict[str, str]:
    """Build a synthetic weight_map mimicking Qwen3.5 tensor naming."""
    weight_map: dict[str, str] = {}

    weight_map["model.embed_tokens.weight"] = "model-00001-of-00004.safetensors"

    tensors_per_layer = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    for layer_idx in range(num_layers):
        shard_num = (layer_idx // 2) + 1
        shard_file = f"model-{shard_num:05d}-of-00004.safetensors"
        for tensor_suffix in tensors_per_layer:
            tensor_name = f"model.layers.{layer_idx}.{tensor_suffix}"
            weight_map[tensor_name] = shard_file

    weight_map["model.norm.weight"] = "model-00004-of-00004.safetensors"
    weight_map["lm_head.weight"] = "model-00004-of-00004.safetensors"

    return weight_map


class _MockModelConfig:
    """Mock model config for testing."""

    def __init__(
        self,
        layer_types: list[str] | None = None,
    ) -> None:
        self.layer_types = layer_types


# ---------------------------------------------------------------------------
# Tests for validate_pipeline_distribution
# ---------------------------------------------------------------------------


class TestValidatePipelineDistribution:
    """Test strict startup validation of pipeline distribution."""

    def _make_distribution(
        self, layers_per_rank: tuple[int, ...], total: int
    ) -> PipelineLayerDistribution:
        return PipelineLayerDistribution(
            layers_per_rank=layers_per_rank,
            total_layer_count=total,
            rank_count=len(layers_per_rank),
        )

    def test_valid_distribution_passes(self) -> None:
        """A valid distribution with matching weight_map passes validation."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(
            layer_types=["full_attention"] * 8,
        )

        # Should not raise
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_valid_distribution_no_layer_types_passes(self) -> None:
        """A valid distribution without layer_types attribute passes."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=None)

        # Should not raise
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_missing_tensor_fails_startup(self) -> None:
        """Missing layer tensor expected by a rank fails startup."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Remove a tensor that rank 0 expects
        del weight_map["model.layers.0.input_layernorm.weight"]

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Missing tensor names for rank 0"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_missing_embedding_tensor_fails_startup(self) -> None:
        """Missing embedding tensor fails startup for rank 0."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        del weight_map["model.embed_tokens.weight"]

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Missing tensor names for rank 0"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_missing_lm_head_tensor_fails_startup(self) -> None:
        """Missing lm_head tensor fails startup for last rank."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        del weight_map["lm_head.weight"]

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Missing tensor names for rank 3"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_missing_norm_tensor_fails_startup(self) -> None:
        """Missing model.norm.weight fails startup for last rank."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        del weight_map["model.norm.weight"]

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Missing tensor names for rank 3"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_duplicate_owned_tensors_fails_startup(self) -> None:
        """Same tensor claimed by two ranks fails startup."""
        weight_map = _build_synthetic_weight_map(num_layers=8)

        # Create a distribution where layer ranges overlap — this is tricky
        # because PipelineLayerDistribution enforces contiguous ranges.
        # Instead, we test by creating a scenario where the ownership resolver
        # would assign the same tensor to multiple ranks.
        # The simplest way: add a layer tensor with an index that falls in
        # two ranks' ranges. But that's impossible with contiguous ranges.
        #
        # The real scenario for duplicates is if the ownership resolver has
        # a bug or if embedding/lm_head are claimed by multiple ranks.
        # We can test this by subclassing or monkeypatching.
        #
        # For a clean test: we'll directly test with a custom weight_map
        # that has a tensor matching both embedding and layer patterns.
        # Actually, the ownership rules are deterministic, so duplicates
        # can't happen with the current resolver. But the validation still
        # needs to check for it as a safety net.
        #
        # Let's test by temporarily patching resolve_tensor_ownership to
        # return overlapping results.
        import unittest.mock

        original_resolve = _loader_mod.resolve_tensor_ownership

        call_count = [0]

        def _mock_resolve(
            wm: dict[str, str], assignment: PipelineStageAssignment
        ) -> list[str]:
            call_count[0] += 1
            result = original_resolve(wm, assignment)
            # Make rank 1 also claim a tensor from rank 0
            if assignment.rank == 1:
                result = sorted(
                    set(result) | {"model.embed_tokens.weight"}
                )
            return result

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with unittest.mock.patch.object(
            _loader_mod, "resolve_tensor_ownership", _mock_resolve
        ):
            with pytest.raises(ValueError, match="Duplicate owned tensors"):
                validate_pipeline_distribution(
                    weight_map=weight_map,
                    layer_distribution=distribution,
                    model_config=config,
                )

    def test_unassigned_layer_tensor_fails_startup(self) -> None:
        """Layer tensor in weight_map not claimed by any rank fails startup."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Add a layer tensor for a layer index outside the distribution range
        # The distribution covers layers 0-7, so layer 8 is unassigned
        weight_map["model.layers.8.self_attn.q_proj.weight"] = (
            "model-00005-of-00004.safetensors"
        )

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Unassigned layer tensors"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_unassigned_embedding_tensor_not_flagged_when_owned(self) -> None:
        """Embedding tensor is assigned to rank 0 and not flagged."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        # Should not raise — embedding is assigned to rank 0
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_metadata_tensors_not_flagged_as_unassigned(self) -> None:
        """Metadata tensors that don't match model patterns are acceptable."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Add a metadata tensor that doesn't match any ownership pattern
        weight_map["__metadata__"] = "model-00001-of-00004.safetensors"
        weight_map["some_random_metadata.info"] = (
            "model-00001-of-00004.safetensors"
        )

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        # Should not raise — metadata tensors are acceptable
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_unsupported_layer_type_fails_startup(self) -> None:
        """Invalid layer type in model config fails startup."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)

        # Include an unsupported layer type
        layer_types = ["full_attention"] * 8
        layer_types[3] = "sliding_window_attention"

        config = _MockModelConfig(layer_types=layer_types)

        with pytest.raises(ValueError, match="Unsupported layer types"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )

    def test_multiple_unsupported_layer_types_reported(self) -> None:
        """Multiple invalid layer types are all reported in the error."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)

        layer_types = ["full_attention"] * 8
        layer_types[2] = "mamba"
        layer_types[5] = "sliding_window"

        config = _MockModelConfig(layer_types=layer_types)

        with pytest.raises(ValueError, match="Unsupported layer types") as exc_info:
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )
        # Both invalid types should be mentioned
        assert "layer 2" in str(exc_info.value)
        assert "layer 5" in str(exc_info.value)

    def test_linear_attention_is_valid(self) -> None:
        """linear_attention is a valid layer type."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((2, 2, 2, 2), 8)

        # Mix of valid types
        layer_types = [
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        config = _MockModelConfig(layer_types=layer_types)

        # Should not raise
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_nonuniform_distribution_passes(self) -> None:
        """Non-uniform distribution with valid weight_map passes."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        distribution = self._make_distribution((3, 2, 2, 1), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        # Should not raise
        validate_pipeline_distribution(
            weight_map=weight_map,
            layer_distribution=distribution,
            model_config=config,
        )

    def test_missing_middle_layer_tensor_fails(self) -> None:
        """Missing tensor for a middle rank's layer fails startup."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Remove a tensor that rank 2 expects (layers 4-5)
        del weight_map["model.layers.4.input_layernorm.weight"]

        distribution = self._make_distribution((2, 2, 2, 2), 8)
        config = _MockModelConfig(layer_types=["full_attention"] * 8)

        with pytest.raises(ValueError, match="Missing tensor names for rank 2"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=config,
            )
