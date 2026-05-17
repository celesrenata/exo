"""
End-to-end integration tests for local-shard model loading.

Creates a synthetic safetensors checkpoint with actual tensor data
(embedding, layers, norm, lm_head), runs the full pipeline for each rank,
verifies each rank loads ONLY its owned tensors, and verifies that a
sequential forward pass through all 4 shards produces the same result
as a monolithic forward pass through all layers.

**Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.12**

Requires both ``torch`` and ``safetensors`` — uses pytest.importorskip.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip if torch or safetensors unavailable
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
safetensors_torch = pytest.importorskip("safetensors.torch")

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
parse_safetensors_index = _loader_mod.parse_safetensors_index
resolve_tensor_ownership = _loader_mod.resolve_tensor_ownership
load_qwen_local_shard_from_safetensors = (
    _loader_mod.load_qwen_local_shard_from_safetensors
)
load_tensors_from_manifest = _loader_mod.load_tensors_from_manifest
validate_loaded_tensors = _loader_mod.validate_loaded_tensors
build_local_shard_modules = _loader_mod.build_local_shard_modules
validate_pipeline_distribution = _loader_mod.validate_pipeline_distribution


# ---------------------------------------------------------------------------
# Model parameters for the synthetic checkpoint
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
HIDDEN_SIZE = 32
VOCAB_SIZE = 100
NUM_RANKS = 4
LAYERS_PER_RANK = (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Fixture: create a complete synthetic safetensors checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_checkpoint(tmp_path: Path) -> Path:
    """Create a complete synthetic safetensors checkpoint.

    Model structure (4 layers, hidden_size=32, vocab_size=100):
    - model.embed_tokens.weight: [100, 32]
    - model.layers.{0..3}.self_attn.q_proj.weight: [32, 32]
    - model.layers.{0..3}.self_attn.k_proj.weight: [32, 32]
    - model.layers.{0..3}.self_attn.v_proj.weight: [32, 32]
    - model.layers.{0..3}.self_attn.o_proj.weight: [32, 32]
    - model.layers.{0..3}.mlp.gate_proj.weight: [32, 32]
    - model.layers.{0..3}.mlp.up_proj.weight: [32, 32]
    - model.layers.{0..3}.mlp.down_proj.weight: [32, 32]
    - model.layers.{0..3}.input_layernorm.weight: [32]
    - model.layers.{0..3}.post_attention_layernorm.weight: [32]
    - model.norm.weight: [32]
    - lm_head.weight: [100, 32]

    Uses deterministic random seed for reproducibility.
    """
    torch.manual_seed(42)

    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}

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

    # Shard 1: embedding + layers 0-1
    shard1 = "model-00001-of-00002.safetensors"
    shard_tensors[shard1] = {}

    embed_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float32)
    shard_tensors[shard1]["model.embed_tokens.weight"] = embed_weight
    weight_map["model.embed_tokens.weight"] = shard1

    for layer_idx in range(2):
        for suffix in layer_suffixes:
            tensor_name = f"model.layers.{layer_idx}.{suffix}"
            if "layernorm" in suffix:
                tensor = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
            else:
                tensor = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float32)
            shard_tensors[shard1][tensor_name] = tensor
            weight_map[tensor_name] = shard1

    # Shard 2: layers 2-3 + norm + lm_head
    shard2 = "model-00002-of-00002.safetensors"
    shard_tensors[shard2] = {}

    for layer_idx in range(2, NUM_LAYERS):
        for suffix in layer_suffixes:
            tensor_name = f"model.layers.{layer_idx}.{suffix}"
            if "layernorm" in suffix:
                tensor = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
            else:
                tensor = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float32)
            shard_tensors[shard2][tensor_name] = tensor
            weight_map[tensor_name] = shard2

    norm_weight = torch.randn(HIDDEN_SIZE, dtype=torch.float32)
    shard_tensors[shard2]["model.norm.weight"] = norm_weight
    weight_map["model.norm.weight"] = shard2

    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float32)
    shard_tensors[shard2]["lm_head.weight"] = lm_head_weight
    weight_map["lm_head.weight"] = shard2

    # Write safetensors files
    for filename, tensors in shard_tensors.items():
        safetensors_torch.save_file(tensors, str(tmp_path / filename))

    # Write the index file
    index_data = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map,
    }
    index_path = tmp_path / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Helper: layer distribution for 4 ranks, 1 layer each
# ---------------------------------------------------------------------------


def _make_distribution() -> Any:
    """Create a 4-rank distribution with 1 layer per rank."""
    return PipelineLayerDistribution(
        layers_per_rank=LAYERS_PER_RANK,
        total_layer_count=NUM_LAYERS,
        rank_count=NUM_RANKS,
    )


# ---------------------------------------------------------------------------
# Tests: each rank loads only its owned tensors
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEachRankLoadsOnlyOwnedTensors:
    """Verify that each rank loads ONLY its assigned tensors from the checkpoint."""

    def test_rank_0_loads_embedding_and_layer_0_only(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Rank 0 loads embedding + layer 0 tensors, nothing else."""
        distribution = _make_distribution()
        assignment = distribution.get_stage_assignment(0)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=synthetic_checkpoint,
            stage_assignment=assignment,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=synthetic_checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Should have embedding + 9 layer tensors = 10
        assert len(loaded) == 10
        assert "model.embed_tokens.weight" in loaded
        assert "model.layers.0.self_attn.q_proj.weight" in loaded
        assert "model.layers.0.input_layernorm.weight" in loaded

        # Should NOT have any other rank's tensors
        assert "model.layers.1.self_attn.q_proj.weight" not in loaded
        assert "model.layers.2.self_attn.q_proj.weight" not in loaded
        assert "model.layers.3.self_attn.q_proj.weight" not in loaded
        assert "model.norm.weight" not in loaded
        assert "lm_head.weight" not in loaded

    def test_rank_1_loads_layer_1_only(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Rank 1 loads only layer 1 tensors."""
        distribution = _make_distribution()
        assignment = distribution.get_stage_assignment(1)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=synthetic_checkpoint,
            stage_assignment=assignment,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=synthetic_checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Should have 9 layer tensors only (no embedding, no norm, no lm_head)
        assert len(loaded) == 9
        assert "model.layers.1.self_attn.q_proj.weight" in loaded
        assert "model.layers.1.mlp.gate_proj.weight" in loaded

        # Should NOT have other ranks' tensors
        assert "model.embed_tokens.weight" not in loaded
        assert "model.layers.0.self_attn.q_proj.weight" not in loaded
        assert "model.layers.2.self_attn.q_proj.weight" not in loaded
        assert "model.norm.weight" not in loaded
        assert "lm_head.weight" not in loaded

    def test_rank_2_loads_layer_2_only(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Rank 2 loads only layer 2 tensors."""
        distribution = _make_distribution()
        assignment = distribution.get_stage_assignment(2)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=synthetic_checkpoint,
            stage_assignment=assignment,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=synthetic_checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Should have 9 layer tensors only
        assert len(loaded) == 9
        assert "model.layers.2.self_attn.q_proj.weight" in loaded
        assert "model.layers.2.mlp.down_proj.weight" in loaded

        # Should NOT have other ranks' tensors
        assert "model.embed_tokens.weight" not in loaded
        assert "model.layers.1.self_attn.q_proj.weight" not in loaded
        assert "model.layers.3.self_attn.q_proj.weight" not in loaded
        assert "model.norm.weight" not in loaded
        assert "lm_head.weight" not in loaded

    def test_rank_3_loads_layer_3_norm_and_lm_head(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Rank 3 loads layer 3 + norm + lm_head tensors."""
        distribution = _make_distribution()
        assignment = distribution.get_stage_assignment(3)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=synthetic_checkpoint,
            stage_assignment=assignment,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=synthetic_checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Should have 9 layer tensors + norm + lm_head = 11
        assert len(loaded) == 11
        assert "model.layers.3.self_attn.q_proj.weight" in loaded
        assert "model.layers.3.mlp.gate_proj.weight" in loaded
        assert "model.norm.weight" in loaded
        assert "lm_head.weight" in loaded

        # Should NOT have other ranks' tensors
        assert "model.embed_tokens.weight" not in loaded
        assert "model.layers.0.self_attn.q_proj.weight" not in loaded
        assert "model.layers.1.self_attn.q_proj.weight" not in loaded
        assert "model.layers.2.self_attn.q_proj.weight" not in loaded

    def test_all_ranks_combined_cover_full_model(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Union of all ranks' loaded tensors equals the full weight map."""
        distribution = _make_distribution()
        weight_map = parse_safetensors_index(synthetic_checkpoint)

        all_loaded_names: set[str] = set()
        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            all_loaded_names.update(loaded.keys())

        assert all_loaded_names == set(weight_map.keys())

    def test_no_tensor_loaded_by_multiple_ranks(
        self, synthetic_checkpoint: Path
    ) -> None:
        """No tensor is loaded by more than one rank."""
        distribution = _make_distribution()
        seen: dict[str, int] = {}

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for tensor_name in loaded:
                assert tensor_name not in seen, (
                    f"Tensor '{tensor_name}' loaded by both rank "
                    f"{seen[tensor_name]} and rank {rank}"
                )
                seen[tensor_name] = rank

    def test_tensor_counts_per_rank(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Verify expected tensor counts for each rank."""
        distribution = _make_distribution()
        expected_counts = {
            0: 10,  # 1 embedding + 9 layer tensors
            1: 9,   # 9 layer tensors
            2: 9,   # 9 layer tensors
            3: 11,  # 9 layer tensors + norm + lm_head
        }

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            assert len(loaded) == expected_counts[rank], (
                f"Rank {rank}: expected {expected_counts[rank]} tensors, "
                f"got {len(loaded)}"
            )


# ---------------------------------------------------------------------------
# Tests: invalid distribution fails
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestInvalidDistributionFails:
    """Verify that invalid pipeline distributions produce clear errors."""

    def test_distribution_sum_mismatch_rejected(self) -> None:
        """Distribution where layers_per_rank sum != total_layer_count is rejected."""
        with pytest.raises(ValueError, match="sums to"):
            PipelineLayerDistribution(
                layers_per_rank=(1, 1, 1, 2),
                total_layer_count=NUM_LAYERS,
                rank_count=NUM_RANKS,
            )

    def test_zero_layer_rank_rejected(self) -> None:
        """Distribution with a zero-layer rank is rejected."""
        with pytest.raises(ValueError, match="at least one layer"):
            PipelineLayerDistribution(
                layers_per_rank=(2, 0, 1, 1),
                total_layer_count=NUM_LAYERS,
                rank_count=NUM_RANKS,
            )

    def test_rank_count_mismatch_rejected(self) -> None:
        """Distribution where len(layers_per_rank) != rank_count is rejected."""
        with pytest.raises(ValueError, match="elements"):
            PipelineLayerDistribution(
                layers_per_rank=(2, 2),
                total_layer_count=NUM_LAYERS,
                rank_count=NUM_RANKS,
            )

    def test_missing_tensor_in_checkpoint_fails(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Validation fails when a required tensor is missing from checkpoint."""
        # Corrupt the index by removing a tensor
        index_path = synthetic_checkpoint / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        del index_data["weight_map"]["model.layers.0.input_layernorm.weight"]

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)

        distribution = _make_distribution()

        class _MockConfig:
            layer_types = ["full_attention"] * NUM_LAYERS

        weight_map = parse_safetensors_index(synthetic_checkpoint)
        with pytest.raises(ValueError, match="Missing tensor names"):
            validate_pipeline_distribution(
                weight_map=weight_map,
                layer_distribution=distribution,
                model_config=_MockConfig(),
            )


# ---------------------------------------------------------------------------
# Tests: toy model pipeline output equals monolithic output
# ---------------------------------------------------------------------------


class _ToyRMSNorm(torch.nn.Module):
    """Minimal RMSNorm for the toy model."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class _ToyTransformerLayer(torch.nn.Module):
    """A toy transformer layer: linear projection + residual + layernorm.

    This is NOT a real transformer layer — it's a deterministic linear
    transform used to verify that pipeline-parallel execution produces
    the same output as monolithic execution.
    """

    def __init__(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        o_proj: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        input_layernorm: torch.Tensor,
        post_attention_layernorm: torch.Tensor,
    ) -> None:
        super().__init__()
        # Simplified: just apply a linear transform + residual
        self.linear = torch.nn.Linear(
            q_proj.shape[1], q_proj.shape[0], bias=False
        )
        self.linear.weight = torch.nn.Parameter(q_proj, requires_grad=False)
        self.norm = _ToyRMSNorm(input_layernorm)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Simplified forward: norm -> linear -> residual
        normed = self.norm(x)
        return x + self.linear(normed)


def _build_toy_layer_from_tensors(
    loaded_tensors: dict[str, torch.Tensor],
    layer_idx: int,
) -> _ToyTransformerLayer:
    """Build a toy transformer layer from loaded tensors for a given layer index."""
    prefix = f"model.layers.{layer_idx}."
    return _ToyTransformerLayer(
        q_proj=loaded_tensors[f"{prefix}self_attn.q_proj.weight"],
        k_proj=loaded_tensors[f"{prefix}self_attn.k_proj.weight"],
        v_proj=loaded_tensors[f"{prefix}self_attn.v_proj.weight"],
        o_proj=loaded_tensors[f"{prefix}self_attn.o_proj.weight"],
        gate_proj=loaded_tensors[f"{prefix}mlp.gate_proj.weight"],
        up_proj=loaded_tensors[f"{prefix}mlp.up_proj.weight"],
        down_proj=loaded_tensors[f"{prefix}mlp.down_proj.weight"],
        input_layernorm=loaded_tensors[f"{prefix}input_layernorm.weight"],
        post_attention_layernorm=loaded_tensors[
            f"{prefix}post_attention_layernorm.weight"
        ],
    )


def _monolithic_forward(
    token_ids: torch.Tensor,
    all_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run a monolithic forward pass through all layers using loaded tensors.

    Steps:
    1. Embed token_ids using model.embed_tokens.weight
    2. Pass through all 4 layers sequentially
    3. Apply final RMSNorm
    4. Apply lm_head linear projection
    5. Return logits
    """
    # Embedding lookup
    embed_weight = all_tensors["model.embed_tokens.weight"]
    hidden_states = torch.nn.functional.embedding(token_ids, embed_weight)

    # Forward through all layers
    for layer_idx in range(NUM_LAYERS):
        layer = _build_toy_layer_from_tensors(all_tensors, layer_idx)
        hidden_states = layer(hidden_states)

    # Final norm
    norm = _ToyRMSNorm(all_tensors["model.norm.weight"])
    hidden_states = norm(hidden_states)

    # lm_head
    lm_head_weight = all_tensors["lm_head.weight"]
    logits = torch.nn.functional.linear(hidden_states, lm_head_weight)

    return logits


def _pipeline_forward(
    token_ids: torch.Tensor,
    rank_tensors: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Run a pipeline-parallel forward pass through all 4 shards sequentially.

    Simulates the pipeline by passing hidden states from rank to rank:
    1. Rank 0: embed -> layer 0 -> output hidden states
    2. Rank 1: layer 1 -> output hidden states
    3. Rank 2: layer 2 -> output hidden states
    4. Rank 3: layer 3 -> norm -> lm_head -> logits
    """
    # Rank 0: embedding + layer 0
    embed_weight = rank_tensors[0]["model.embed_tokens.weight"]
    hidden_states = torch.nn.functional.embedding(token_ids, embed_weight)
    layer_0 = _build_toy_layer_from_tensors(rank_tensors[0], 0)
    hidden_states = layer_0(hidden_states)

    # Rank 1: layer 1
    layer_1 = _build_toy_layer_from_tensors(rank_tensors[1], 1)
    hidden_states = layer_1(hidden_states)

    # Rank 2: layer 2
    layer_2 = _build_toy_layer_from_tensors(rank_tensors[2], 2)
    hidden_states = layer_2(hidden_states)

    # Rank 3: layer 3 + norm + lm_head
    layer_3 = _build_toy_layer_from_tensors(rank_tensors[3], 3)
    hidden_states = layer_3(hidden_states)

    norm = _ToyRMSNorm(rank_tensors[3]["model.norm.weight"])
    hidden_states = norm(hidden_states)

    lm_head_weight = rank_tensors[3]["lm_head.weight"]
    logits = torch.nn.functional.linear(hidden_states, lm_head_weight)

    return logits


@pytest.mark.slow
class TestPipelineOutputEqualsMonolithic:
    """Verify that pipeline-parallel forward pass equals monolithic forward pass."""

    def test_pipeline_equals_monolithic_single_token(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Single token: pipeline output matches monolithic output exactly."""
        distribution = _make_distribution()
        device = torch.device("cpu")
        dtype = torch.float32

        # Load all tensors monolithically (all ranks combined)
        all_tensors: dict[str, torch.Tensor] = {}
        rank_tensors: list[dict[str, torch.Tensor]] = []

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=device,
                dtype=dtype,
            )
            rank_tensors.append(loaded)
            all_tensors.update(loaded)

        # Input: single token
        token_ids = torch.tensor([[42]], dtype=torch.long)

        # Monolithic forward
        monolithic_logits = _monolithic_forward(token_ids, all_tensors)

        # Pipeline forward
        pipeline_logits = _pipeline_forward(token_ids, rank_tensors)

        # They must be exactly equal (same tensors, same operations, float32)
        assert torch.allclose(
            monolithic_logits, pipeline_logits, rtol=1e-5, atol=1e-5
        ), (
            f"Pipeline output differs from monolithic.\n"
            f"Max diff: {(monolithic_logits - pipeline_logits).abs().max().item()}"
        )

    def test_pipeline_equals_monolithic_sequence(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Short sequence: pipeline output matches monolithic output."""
        distribution = _make_distribution()
        device = torch.device("cpu")
        dtype = torch.float32

        # Load tensors for each rank
        all_tensors: dict[str, torch.Tensor] = {}
        rank_tensors: list[dict[str, torch.Tensor]] = []

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=device,
                dtype=dtype,
            )
            rank_tensors.append(loaded)
            all_tensors.update(loaded)

        # Input: sequence of 5 tokens
        token_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

        # Monolithic forward
        monolithic_logits = _monolithic_forward(token_ids, all_tensors)

        # Pipeline forward
        pipeline_logits = _pipeline_forward(token_ids, rank_tensors)

        assert torch.allclose(
            monolithic_logits, pipeline_logits, rtol=1e-5, atol=1e-5
        ), (
            f"Pipeline output differs from monolithic for sequence.\n"
            f"Max diff: {(monolithic_logits - pipeline_logits).abs().max().item()}"
        )

        # Verify output shape: [batch=1, seq_len=5, vocab_size=100]
        assert monolithic_logits.shape == (1, 5, VOCAB_SIZE)
        assert pipeline_logits.shape == (1, 5, VOCAB_SIZE)

    def test_pipeline_equals_monolithic_batch(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Batch of sequences: pipeline output matches monolithic output."""
        distribution = _make_distribution()
        device = torch.device("cpu")
        dtype = torch.float32

        # Load tensors for each rank
        all_tensors: dict[str, torch.Tensor] = {}
        rank_tensors: list[dict[str, torch.Tensor]] = []

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=device,
                dtype=dtype,
            )
            rank_tensors.append(loaded)
            all_tensors.update(loaded)

        # Input: batch of 2 sequences, each 3 tokens
        token_ids = torch.tensor(
            [[5, 15, 25], [35, 45, 55]], dtype=torch.long
        )

        # Monolithic forward
        monolithic_logits = _monolithic_forward(token_ids, all_tensors)

        # Pipeline forward
        pipeline_logits = _pipeline_forward(token_ids, rank_tensors)

        assert torch.allclose(
            monolithic_logits, pipeline_logits, rtol=1e-5, atol=1e-5
        ), (
            f"Pipeline output differs from monolithic for batch.\n"
            f"Max diff: {(monolithic_logits - pipeline_logits).abs().max().item()}"
        )

        # Verify output shape: [batch=2, seq_len=3, vocab_size=100]
        assert monolithic_logits.shape == (2, 3, VOCAB_SIZE)
        assert pipeline_logits.shape == (2, 3, VOCAB_SIZE)

    def test_argmax_token_selection_matches(
        self, synthetic_checkpoint: Path
    ) -> None:
        """Greedy token selection from pipeline matches monolithic."""
        distribution = _make_distribution()
        device = torch.device("cpu")
        dtype = torch.float32

        all_tensors: dict[str, torch.Tensor] = {}
        rank_tensors: list[dict[str, torch.Tensor]] = []

        for rank in range(NUM_RANKS):
            assignment = distribution.get_stage_assignment(rank)
            manifest = load_qwen_local_shard_from_safetensors(
                model_path=synthetic_checkpoint,
                stage_assignment=assignment,
            )
            loaded = load_tensors_from_manifest(
                manifest=manifest,
                model_path=synthetic_checkpoint,
                device=device,
                dtype=dtype,
            )
            rank_tensors.append(loaded)
            all_tensors.update(loaded)

        # Input: single token for next-token prediction
        token_ids = torch.tensor([[7]], dtype=torch.long)

        monolithic_logits = _monolithic_forward(token_ids, all_tensors)
        pipeline_logits = _pipeline_forward(token_ids, rank_tensors)

        # Greedy selection: argmax over vocabulary dimension
        monolithic_token = torch.argmax(monolithic_logits[:, -1, :], dim=-1)
        pipeline_token = torch.argmax(pipeline_logits[:, -1, :], dim=-1)

        assert torch.equal(monolithic_token, pipeline_token), (
            f"Greedy token mismatch: monolithic={monolithic_token.item()}, "
            f"pipeline={pipeline_token.item()}"
        )
