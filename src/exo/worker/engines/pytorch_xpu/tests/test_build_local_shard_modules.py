"""
Unit tests for build_local_shard_modules.

Tests that the function correctly instantiates local layers using a mock
layer builder, preserves global layer indices, and creates embedding/lm_head/norm
only for the correct ranks.

**Validates: Requirements 1.1, 1.7**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
LocalShardModules = _loader_mod.LocalShardModules
build_local_shard_modules = _loader_mod.build_local_shard_modules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 32
VOCAB_SIZE = 100
NUM_LAYERS = 8
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


class _MockModelConfig:
    """Mock model config mimicking Qwen3.5 with layer_types metadata."""

    def __init__(self, num_layers: int = NUM_LAYERS) -> None:
        # Qwen3.5 pattern: first 25% full_attention, rest linear_attention
        self.layer_types: list[str] = []
        for i in range(num_layers):
            if i % 4 == 0:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("linear_attention")
        self.hidden_size = HIDDEN_SIZE
        self.num_attention_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.rope_theta = 10000.0
        self.max_position_embeddings = 4096
        self.partial_rotary_factor = 1.0


def _mock_layer_builder(
    *,
    layer_type: str,
    global_layer_index: int,
    layer_tensors: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    model_config: object,
) -> torch.nn.Module:
    """Mock layer builder that creates a simple Linear module."""
    module = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
    # Tag the module with metadata for test assertions
    module._test_layer_type = layer_type  # type: ignore[attr-defined]
    module._test_global_index = global_layer_index  # type: ignore[attr-defined]
    module._test_tensor_count = len(layer_tensors)  # type: ignore[attr-defined]
    return module.to(device=device, dtype=dtype)


def _build_synthetic_tensors(
    num_layers: int = NUM_LAYERS,
    include_embedding: bool = True,
    include_lm_head: bool = True,
    include_norm: bool = True,
) -> dict[str, torch.Tensor]:
    """Build synthetic loaded tensors mimicking Qwen3.5 naming."""
    tensors: dict[str, torch.Tensor] = {}

    if include_embedding:
        tensors["model.embed_tokens.weight"] = torch.randn(
            VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16
        )

    layer_suffixes = [
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
        for suffix in layer_suffixes:
            tensor_name = f"model.layers.{layer_idx}.{suffix}"
            if "layernorm" in suffix:
                tensors[tensor_name] = torch.randn(
                    HIDDEN_SIZE, dtype=torch.bfloat16
                )
            else:
                tensors[tensor_name] = torch.randn(
                    HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16
                )

    if include_norm:
        tensors["model.norm.weight"] = torch.randn(
            HIDDEN_SIZE, dtype=torch.bfloat16
        )

    if include_lm_head:
        tensors["lm_head.weight"] = torch.randn(
            VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16
        )

    return tensors


def _get_assignment(rank: int, num_layers: int = NUM_LAYERS) -> PipelineStageAssignment:
    """Get stage assignment for a balanced distribution."""
    layers_per_rank = num_layers // 4
    distribution = PipelineLayerDistribution(
        layers_per_rank=(layers_per_rank,) * 4,
        total_layer_count=num_layers,
        rank_count=4,
    )
    return distribution.get_stage_assignment(rank)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildLocalShardModulesLayerCount:
    """Verify correct number of layers are created."""

    def test_creates_correct_number_of_layers_rank_0(self) -> None:
        """Rank 0 gets exactly its assigned number of layers."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert len(result.layers) == assignment.num_local_layers
        assert len(result.layers) == 2  # 8 layers / 4 ranks = 2

    def test_creates_correct_number_of_layers_rank_1(self) -> None:
        """Middle rank gets exactly its assigned number of layers."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(1)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert len(result.layers) == 2

    def test_creates_correct_number_of_layers_last_rank(self) -> None:
        """Last rank gets exactly its assigned number of layers."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert len(result.layers) == 2


class TestBuildLocalShardModulesGlobalIndices:
    """Verify global layer indices are preserved."""

    def test_rank_0_has_indices_0_1(self) -> None:
        """Rank 0 preserves global indices [0, 1]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.global_layer_indices == [0, 1]

    def test_rank_1_has_indices_2_3(self) -> None:
        """Rank 1 preserves global indices [2, 3]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(1)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.global_layer_indices == [2, 3]

    def test_rank_2_has_indices_4_5(self) -> None:
        """Rank 2 preserves global indices [4, 5]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(2)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.global_layer_indices == [4, 5]

    def test_rank_3_has_indices_6_7(self) -> None:
        """Rank 3 preserves global indices [6, 7]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.global_layer_indices == [6, 7]

    def test_layer_builder_receives_correct_global_index(self) -> None:
        """Layer builder is called with the correct global index."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(2)  # layers [4, 6)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        # Check the tagged metadata on mock layers
        for i, layer in enumerate(result.layers):
            assert layer._test_global_index == 4 + i  # type: ignore[attr-defined]


class TestBuildLocalShardModulesEmbedding:
    """Verify embedding is only created for rank 0."""

    def test_rank_0_has_embedding(self) -> None:
        """Rank 0 creates the embedding module."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.embed_tokens is not None
        assert isinstance(result.embed_tokens, torch.nn.Embedding)
        assert result.embed_tokens.num_embeddings == VOCAB_SIZE
        assert result.embed_tokens.embedding_dim == HIDDEN_SIZE

    def test_rank_1_has_no_embedding(self) -> None:
        """Middle rank does not create embedding."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(1)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.embed_tokens is None

    def test_rank_3_has_no_embedding(self) -> None:
        """Last rank does not create embedding."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.embed_tokens is None


class TestBuildLocalShardModulesLmHead:
    """Verify lm_head and final_norm are only created for last rank."""

    def test_last_rank_has_lm_head(self) -> None:
        """Last rank creates lm_head."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.lm_head is not None
        assert isinstance(result.lm_head, torch.nn.Linear)
        assert result.lm_head.in_features == HIDDEN_SIZE
        assert result.lm_head.out_features == VOCAB_SIZE

    def test_last_rank_has_final_norm(self) -> None:
        """Last rank creates final_norm."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.final_norm is not None

    def test_rank_0_has_no_lm_head(self) -> None:
        """Rank 0 does not create lm_head."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.lm_head is None
        assert result.final_norm is None

    def test_middle_rank_has_no_lm_head(self) -> None:
        """Middle rank does not create lm_head or final_norm."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(2)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.lm_head is None
        assert result.final_norm is None


class TestBuildLocalShardModulesLayerTypes:
    """Verify layer_types list matches model config for the local range."""

    def test_layer_types_match_config_rank_0(self) -> None:
        """Rank 0 layer types match model_config.layer_types[0:2]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        expected = config.layer_types[0:2]
        assert result.layer_types == expected

    def test_layer_types_match_config_rank_1(self) -> None:
        """Rank 1 layer types match model_config.layer_types[2:4]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(1)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        expected = config.layer_types[2:4]
        assert result.layer_types == expected

    def test_layer_types_match_config_rank_3(self) -> None:
        """Last rank layer types match model_config.layer_types[6:8]."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        expected = config.layer_types[6:8]
        assert result.layer_types == expected

    def test_layer_builder_receives_correct_type(self) -> None:
        """Layer builder is called with the correct layer type."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(1)  # layers [2, 4)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        for i, layer in enumerate(result.layers):
            expected_type = config.layer_types[2 + i]
            assert layer._test_layer_type == expected_type  # type: ignore[attr-defined]

    def test_defaults_to_full_attention_without_layer_types(self) -> None:
        """Without layer_types in config, defaults to full_attention."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = MagicMock()
        config.layer_types = None  # Simulate missing attribute
        del config.layer_types  # Make getattr return None
        config.hidden_size = HIDDEN_SIZE
        config.num_attention_heads = NUM_HEADS
        config.head_dim = HEAD_DIM
        config.rope_theta = 10000.0
        config.max_position_embeddings = 4096
        config.partial_rotary_factor = 1.0

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert all(lt == "full_attention" for lt in result.layer_types)


class TestBuildLocalShardModulesRotaryEmbedding:
    """Verify rotary embedding is created for all ranks."""

    def test_rank_0_has_rotary_emb(self) -> None:
        """Rank 0 gets a rotary embedding module."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.rotary_emb is not None

    def test_middle_rank_has_rotary_emb(self) -> None:
        """Middle rank gets a rotary embedding module."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(2)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.rotary_emb is not None

    def test_last_rank_has_rotary_emb(self) -> None:
        """Last rank gets a rotary embedding module."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert result.rotary_emb is not None


class TestBuildLocalShardModulesValidation:
    """Verify error handling for invalid inputs."""

    def test_raises_on_invalid_layer_type(self) -> None:
        """Invalid layer type in config raises ValueError."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)
        config = _MockModelConfig()
        config.layer_types[0] = "invalid_type"

        with pytest.raises(ValueError, match="Invalid layer type"):
            build_local_shard_modules(
                loaded_tensors=tensors,
                stage_assignment=assignment,
                model_config=config,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                layer_builder=_mock_layer_builder,
            )

    def test_raises_on_insufficient_layer_types(self) -> None:
        """Config with too few layer_types raises ValueError."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(3)  # needs layers up to index 7
        config = _MockModelConfig()
        config.layer_types = ["full_attention"] * 4  # Only 4, need 8

        with pytest.raises(ValueError, match="layer_types has 4 entries"):
            build_local_shard_modules(
                loaded_tensors=tensors,
                stage_assignment=assignment,
                model_config=config,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                layer_builder=_mock_layer_builder,
            )

    def test_layer_tensors_passed_to_builder(self) -> None:
        """Layer builder receives the correct tensors for each layer."""
        tensors = _build_synthetic_tensors()
        assignment = _get_assignment(0)  # layers [0, 2)
        config = _MockModelConfig()

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        # Each layer should have received 9 tensors (the layer suffixes)
        for layer in result.layers:
            assert layer._test_tensor_count == 9  # type: ignore[attr-defined]


class TestBuildLocalShardModulesNonUniformDistribution:
    """Verify behavior with non-uniform layer distributions."""

    def test_nonuniform_distribution(self) -> None:
        """Non-uniform distribution creates correct layer counts."""
        num_layers = 8
        tensors = _build_synthetic_tensors(num_layers=num_layers)
        # [3, 2, 2, 1] distribution
        distribution = PipelineLayerDistribution(
            layers_per_rank=(3, 2, 2, 1),
            total_layer_count=num_layers,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        config = _MockModelConfig(num_layers=num_layers)

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert len(result.layers) == 3
        assert result.global_layer_indices == [0, 1, 2]

    def test_nonuniform_last_rank_single_layer(self) -> None:
        """Last rank with single layer works correctly."""
        num_layers = 8
        tensors = _build_synthetic_tensors(num_layers=num_layers)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(3, 2, 2, 1),
            total_layer_count=num_layers,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(3)
        config = _MockModelConfig(num_layers=num_layers)

        result = build_local_shard_modules(
            loaded_tensors=tensors,
            stage_assignment=assignment,
            model_config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            layer_builder=_mock_layer_builder,
        )

        assert len(result.layers) == 1
        assert result.global_layer_indices == [7]
        assert result.lm_head is not None
        assert result.final_norm is not None
        assert result.embed_tokens is None
