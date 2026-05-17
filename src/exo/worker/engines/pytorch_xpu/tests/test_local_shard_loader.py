"""
Unit tests for local-shard model loading.

Tests safetensors index parsing, tensor ownership resolution,
and local tensor manifest creation for Qwen3.5 pipeline-parallel loading.

**Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.9, 1.10**
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
#
# We must ensure that local_shard_loader.py uses the SAME PipelineStageAssignment
# class as the test. We achieve this by pre-loading pipeline_config under its
# canonical module path so that local_shard_loader's import resolves to it.
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


# Load pipeline_config under its canonical import path so that
# local_shard_loader.py's import resolves to the same class instances.
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


# ---------------------------------------------------------------------------
# Fixtures — synthetic safetensors index for a small Qwen3.5-like model
# ---------------------------------------------------------------------------


def _build_synthetic_weight_map(num_layers: int = 8) -> dict[str, str]:
    """Build a synthetic weight_map mimicking Qwen3.5 tensor naming.

    Creates tensors for:
    - model.embed_tokens.weight
    - model.layers.{i}.self_attn.q_proj.weight
    - model.layers.{i}.self_attn.k_proj.weight
    - model.layers.{i}.self_attn.v_proj.weight
    - model.layers.{i}.self_attn.o_proj.weight
    - model.layers.{i}.mlp.gate_proj.weight
    - model.layers.{i}.mlp.up_proj.weight
    - model.layers.{i}.mlp.down_proj.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.norm.weight
    - lm_head.weight
    """
    weight_map: dict[str, str] = {}

    # Embedding
    weight_map["model.embed_tokens.weight"] = "model-00001-of-00004.safetensors"

    # Layers
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
        # Distribute layers across shard files
        shard_num = (layer_idx // 2) + 1
        shard_file = f"model-{shard_num:05d}-of-00004.safetensors"
        for tensor_suffix in tensors_per_layer:
            tensor_name = f"model.layers.{layer_idx}.{tensor_suffix}"
            weight_map[tensor_name] = shard_file

    # Final norm and lm_head
    weight_map["model.norm.weight"] = "model-00004-of-00004.safetensors"
    weight_map["lm_head.weight"] = "model-00004-of-00004.safetensors"

    return weight_map


def _write_synthetic_index(
    tmp_path: Path, num_layers: int = 8
) -> dict[str, str]:
    """Write a synthetic safetensors index to tmp_path and return weight_map."""
    weight_map = _build_synthetic_weight_map(num_layers)
    index_data = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map,
    }
    index_path = tmp_path / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)
    return weight_map


# ---------------------------------------------------------------------------
# Tests for parse_safetensors_index
# ---------------------------------------------------------------------------


class TestParseSafetensorsIndex:
    """Test safetensors index file parsing."""

    def test_parses_valid_index(self, tmp_path: Path) -> None:
        """A valid index file is parsed correctly."""
        expected_map = _write_synthetic_index(tmp_path, num_layers=4)
        result = parse_safetensors_index(tmp_path)
        assert result == expected_map

    def test_returns_all_tensor_names(self, tmp_path: Path) -> None:
        """Parsed weight_map contains all expected tensor names."""
        _write_synthetic_index(tmp_path, num_layers=4)
        result = parse_safetensors_index(tmp_path)
        # 1 embedding + 4*9 layer tensors + 1 norm + 1 lm_head = 39
        assert len(result) == 39

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing index file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Safetensors index not found"):
            parse_safetensors_index(tmp_path)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Malformed JSON raises ValueError."""
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_safetensors_index(tmp_path)

    def test_raises_on_missing_weight_map(self, tmp_path: Path) -> None:
        """Index without weight_map key raises ValueError."""
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"metadata": {}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no 'weight_map' key"):
            parse_safetensors_index(tmp_path)

    def test_raises_on_non_dict_weight_map(self, tmp_path: Path) -> None:
        """Index with non-dict weight_map raises ValueError."""
        index_path = tmp_path / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"weight_map": ["not", "a", "dict"]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="is not a dict"):
            parse_safetensors_index(tmp_path)


# ---------------------------------------------------------------------------
# Tests for resolve_tensor_ownership
# ---------------------------------------------------------------------------


class TestResolveTensorOwnership:
    """Test tensor ownership resolution for each rank in a 4-rank distribution."""

    def _get_assignment(self, rank: int) -> PipelineStageAssignment:
        """Get stage assignment for a balanced 8-layer, 4-rank distribution."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        return distribution.get_stage_assignment(rank)

    def test_rank_0_owns_embedding(self) -> None:
        """Rank 0 owns model.embed_tokens.weight."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(0)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert "model.embed_tokens.weight" in owned

    def test_rank_0_does_not_own_lm_head(self) -> None:
        """Rank 0 does not own lm_head.weight or model.norm.weight."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(0)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert "lm_head.weight" not in owned
        assert "model.norm.weight" not in owned

    def test_rank_0_owns_layers_0_and_1(self) -> None:
        """Rank 0 owns tensors for layers 0 and 1."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(0)
        owned = resolve_tensor_ownership(weight_map, assignment)
        # Check layer 0 tensors present
        assert "model.layers.0.self_attn.q_proj.weight" in owned
        assert "model.layers.0.mlp.gate_proj.weight" in owned
        # Check layer 1 tensors present
        assert "model.layers.1.self_attn.q_proj.weight" in owned
        # Check layer 2 tensors NOT present
        assert "model.layers.2.self_attn.q_proj.weight" not in owned

    def test_last_rank_owns_norm_and_lm_head(self) -> None:
        """Last rank (rank 3) owns model.norm.weight and lm_head.weight."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(3)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert "model.norm.weight" in owned
        assert "lm_head.weight" in owned

    def test_last_rank_does_not_own_embedding(self) -> None:
        """Last rank does not own model.embed_tokens.weight."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(3)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert "model.embed_tokens.weight" not in owned

    def test_last_rank_owns_layers_6_and_7(self) -> None:
        """Last rank owns tensors for layers 6 and 7."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(3)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert "model.layers.6.self_attn.q_proj.weight" in owned
        assert "model.layers.7.mlp.down_proj.weight" in owned
        # Layer 5 should NOT be owned
        assert "model.layers.5.self_attn.q_proj.weight" not in owned

    def test_middle_rank_owns_only_assigned_layers(self) -> None:
        """Middle ranks own only their assigned layer tensors."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Rank 1 owns layers [2, 4)
        assignment = self._get_assignment(1)
        owned = resolve_tensor_ownership(weight_map, assignment)

        # Should own layers 2 and 3
        assert "model.layers.2.self_attn.q_proj.weight" in owned
        assert "model.layers.3.mlp.gate_proj.weight" in owned

        # Should NOT own embedding, norm, lm_head
        assert "model.embed_tokens.weight" not in owned
        assert "model.norm.weight" not in owned
        assert "lm_head.weight" not in owned

        # Should NOT own layers outside range
        assert "model.layers.1.self_attn.q_proj.weight" not in owned
        assert "model.layers.4.self_attn.q_proj.weight" not in owned

    def test_middle_rank_2_owns_layers_4_and_5(self) -> None:
        """Rank 2 owns layers [4, 6)."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(2)
        owned = resolve_tensor_ownership(weight_map, assignment)

        assert "model.layers.4.self_attn.q_proj.weight" in owned
        assert "model.layers.5.mlp.up_proj.weight" in owned
        assert "model.layers.3.self_attn.q_proj.weight" not in owned
        assert "model.layers.6.self_attn.q_proj.weight" not in owned

    def test_all_ranks_cover_all_tensors(self) -> None:
        """Union of all ranks' owned tensors equals the full weight map."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        all_owned: set[str] = set()
        for rank in range(4):
            assignment = self._get_assignment(rank)
            owned = resolve_tensor_ownership(weight_map, assignment)
            all_owned.update(owned)
        assert all_owned == set(weight_map.keys())

    def test_no_tensor_owned_by_multiple_ranks(self) -> None:
        """No tensor is owned by more than one rank."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        seen: dict[str, int] = {}
        for rank in range(4):
            assignment = self._get_assignment(rank)
            owned = resolve_tensor_ownership(weight_map, assignment)
            for tensor_name in owned:
                assert tensor_name not in seen, (
                    f"Tensor '{tensor_name}' owned by both rank "
                    f"{seen[tensor_name]} and rank {rank}"
                )
                seen[tensor_name] = rank

    def test_owned_tensors_are_sorted(self) -> None:
        """Returned tensor names are sorted."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        assignment = self._get_assignment(0)
        owned = resolve_tensor_ownership(weight_map, assignment)
        assert owned == sorted(owned)

    def test_tied_weights_handled(self) -> None:
        """Tied weights: lm_head.weight tied to embed_tokens still goes to last rank."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # Simulate tied weights — same shard file for both
        weight_map["lm_head.weight"] = weight_map["model.embed_tokens.weight"]

        # Last rank still owns lm_head.weight
        last_assignment = self._get_assignment(3)
        last_owned = resolve_tensor_ownership(weight_map, last_assignment)
        assert "lm_head.weight" in last_owned

        # First rank owns embed_tokens but NOT lm_head
        first_assignment = self._get_assignment(0)
        first_owned = resolve_tensor_ownership(weight_map, first_assignment)
        assert "model.embed_tokens.weight" in first_owned
        assert "lm_head.weight" not in first_owned

    def test_nonuniform_distribution(self) -> None:
        """Non-uniform distribution assigns correct layers."""
        weight_map = _build_synthetic_weight_map(num_layers=8)
        # [3, 2, 2, 1] distribution
        distribution = PipelineLayerDistribution(
            layers_per_rank=(3, 2, 2, 1),
            total_layer_count=8,
            rank_count=4,
        )

        # Rank 0: layers [0, 3)
        a0 = distribution.get_stage_assignment(0)
        owned_0 = resolve_tensor_ownership(weight_map, a0)
        assert "model.layers.0.self_attn.q_proj.weight" in owned_0
        assert "model.layers.2.self_attn.q_proj.weight" in owned_0
        assert "model.layers.3.self_attn.q_proj.weight" not in owned_0

        # Rank 3: layers [7, 8) — only layer 7
        a3 = distribution.get_stage_assignment(3)
        owned_3 = resolve_tensor_ownership(weight_map, a3)
        assert "model.layers.7.self_attn.q_proj.weight" in owned_3
        assert "model.layers.6.self_attn.q_proj.weight" not in owned_3
        assert "lm_head.weight" in owned_3

    def test_nonuniform_produces_different_ownership_than_default(self) -> None:
        """A non-uniform distribution produces different tensor ownership than balanced default.

        This test explicitly demonstrates that the configured distribution
        (not the default balanced one) determines tensor ownership. With 8
        layers and 4 ranks:
        - Default balanced: (2, 2, 2, 2) → rank 1 owns layers [2, 4)
        - Non-uniform:      (3, 2, 2, 1) → rank 1 owns layers [3, 5)

        The ownership sets differ, proving the distribution is respected.
        """
        weight_map = _build_synthetic_weight_map(num_layers=8)

        # Default balanced distribution: (2, 2, 2, 2)
        balanced = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )

        # Non-uniform distribution: (3, 2, 2, 1)
        nonuniform = PipelineLayerDistribution(
            layers_per_rank=(3, 2, 2, 1),
            total_layer_count=8,
            rank_count=4,
        )

        # Compare rank 1 ownership under both distributions
        balanced_a1 = balanced.get_stage_assignment(1)
        nonuniform_a1 = nonuniform.get_stage_assignment(1)

        balanced_owned_1 = set(resolve_tensor_ownership(weight_map, balanced_a1))
        nonuniform_owned_1 = set(resolve_tensor_ownership(weight_map, nonuniform_a1))

        # The two distributions produce different ownership for rank 1
        assert balanced_owned_1 != nonuniform_owned_1, (
            "Non-uniform distribution must produce different tensor ownership "
            "than the default balanced distribution"
        )

        # Balanced rank 1 owns layers [2, 4) — has layer 2 tensors
        assert "model.layers.2.self_attn.q_proj.weight" in balanced_owned_1
        assert "model.layers.3.self_attn.q_proj.weight" in balanced_owned_1

        # Non-uniform rank 1 owns layers [3, 5) — has layer 3 and 4 tensors
        assert "model.layers.3.self_attn.q_proj.weight" in nonuniform_owned_1
        assert "model.layers.4.self_attn.q_proj.weight" in nonuniform_owned_1
        # Non-uniform rank 1 does NOT own layer 2 (that's rank 0's)
        assert "model.layers.2.self_attn.q_proj.weight" not in nonuniform_owned_1

        # Both distributions still cover all tensors (partition property)
        for dist in (balanced, nonuniform):
            all_owned: set[str] = set()
            for rank in range(4):
                assignment = dist.get_stage_assignment(rank)
                owned = resolve_tensor_ownership(weight_map, assignment)
                all_owned.update(owned)
            assert all_owned == set(weight_map.keys())


# ---------------------------------------------------------------------------
# Tests for load_qwen_local_shard_from_safetensors
# ---------------------------------------------------------------------------


class TestLoadQwenLocalShardFromSafetensors:
    """Test the orchestrator function that creates local shard manifests."""

    def test_creates_manifest_for_rank_0(self, tmp_path: Path) -> None:
        """Creates a valid manifest for rank 0."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        assert isinstance(manifest, LocalShardManifest)
        assert manifest.stage_assignment == assignment
        assert "model.embed_tokens.weight" in manifest.tensor_names
        assert manifest.local_tensor_count > 0
        # 1 embedding + 2 layers * 9 tensors = 19
        assert manifest.local_tensor_count == 19

    def test_creates_manifest_for_last_rank(self, tmp_path: Path) -> None:
        """Creates a valid manifest for the last rank."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(3)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        assert "model.norm.weight" in manifest.tensor_names
        assert "lm_head.weight" in manifest.tensor_names
        # 2 layers * 9 tensors + norm + lm_head = 20
        assert manifest.local_tensor_count == 20

    def test_creates_manifest_for_middle_rank(self, tmp_path: Path) -> None:
        """Creates a valid manifest for a middle rank."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(1)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        # Middle rank: 2 layers * 9 tensors = 18
        assert manifest.local_tensor_count == 18
        assert "model.embed_tokens.weight" not in manifest.tensor_names
        assert "model.norm.weight" not in manifest.tensor_names
        assert "lm_head.weight" not in manifest.tensor_names

    def test_total_tensor_count_is_full_model(self, tmp_path: Path) -> None:
        """total_tensor_count reflects the full model tensor count."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        # 1 embed + 8*9 layers + 1 norm + 1 lm_head = 75
        assert manifest.total_tensor_count == 75

    def test_tensor_to_file_maps_correctly(self, tmp_path: Path) -> None:
        """tensor_to_file maps each owned tensor to its shard file."""
        weight_map = _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        for tensor_name in manifest.tensor_names:
            assert tensor_name in manifest.tensor_to_file
            assert manifest.tensor_to_file[tensor_name] == weight_map[tensor_name]

    def test_raises_on_missing_index(self, tmp_path: Path) -> None:
        """Missing index file raises FileNotFoundError."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)

        with pytest.raises(FileNotFoundError):
            load_qwen_local_shard_from_safetensors(
                model_path=tmp_path,
                stage_assignment=assignment,
            )

    def test_manifest_is_immutable(self, tmp_path: Path) -> None:
        """LocalShardManifest instances are immutable (frozen)."""
        from pydantic import ValidationError

        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        with pytest.raises(ValidationError):
            manifest.local_tensor_count = 999  # type: ignore[misc]

    def test_manifest_tensor_names_sorted(self, tmp_path: Path) -> None:
        """Manifest tensor_names are in sorted order."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(2)

        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        assert manifest.tensor_names == sorted(manifest.tensor_names)

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        """Same inputs produce identical manifests across repeated calls."""
        _write_synthetic_index(tmp_path, num_layers=8)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2, 2, 2),
            total_layer_count=8,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(1)

        manifest_1 = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )
        manifest_2 = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path,
            stage_assignment=assignment,
        )

        assert manifest_1.tensor_names == manifest_2.tensor_names
        assert manifest_1.tensor_to_file == manifest_2.tensor_to_file
        assert manifest_1.local_tensor_count == manifest_2.local_tensor_count
        assert manifest_1.total_tensor_count == manifest_2.total_tensor_count



# ---------------------------------------------------------------------------
# Tests for load_tensors_from_manifest and validate_loaded_tensors
# ---------------------------------------------------------------------------

# Import the new functions from the loader module
load_tensors_from_manifest = _loader_mod.load_tensors_from_manifest
validate_loaded_tensors = _loader_mod.validate_loaded_tensors


def _has_safetensors() -> bool:
    """Check if safetensors and torch are available."""
    try:
        import safetensors.torch  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


_requires_safetensors = pytest.mark.skipif(
    not _has_safetensors(),
    reason="safetensors and torch required",
)


def _create_synthetic_safetensors_files(
    tmp_path: Path, num_layers: int = 4
) -> dict[str, str]:
    """Create actual safetensors files with known tensor values.

    Returns the weight_map (tensor_name -> shard_filename).
    """
    import torch
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    hidden_size = 16

    # Group tensors by shard file
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}

    # Embedding — shard 1
    shard_file = "model-00001-of-00002.safetensors"
    embed_tensor = torch.randn(100, hidden_size, dtype=torch.float32)
    if shard_file not in shard_tensors:
        shard_tensors[shard_file] = {}
    shard_tensors[shard_file]["model.embed_tokens.weight"] = embed_tensor
    weight_map["model.embed_tokens.weight"] = shard_file

    # Layer tensors
    layer_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "mlp.gate_proj.weight",
        "input_layernorm.weight",
    ]

    for layer_idx in range(num_layers):
        shard_num = 1 if layer_idx < num_layers // 2 else 2
        shard_file = f"model-{shard_num:05d}-of-00002.safetensors"
        if shard_file not in shard_tensors:
            shard_tensors[shard_file] = {}

        for suffix in layer_suffixes:
            tensor_name = f"model.layers.{layer_idx}.{suffix}"
            if "layernorm" in suffix:
                tensor = torch.randn(hidden_size, dtype=torch.float32)
            else:
                tensor = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
            shard_tensors[shard_file][tensor_name] = tensor
            weight_map[tensor_name] = shard_file

    # Final norm and lm_head — shard 2
    shard_file = "model-00002-of-00002.safetensors"
    if shard_file not in shard_tensors:
        shard_tensors[shard_file] = {}
    shard_tensors[shard_file]["model.norm.weight"] = torch.randn(
        hidden_size, dtype=torch.float32
    )
    weight_map["model.norm.weight"] = shard_file
    shard_tensors[shard_file]["lm_head.weight"] = torch.randn(
        100, hidden_size, dtype=torch.float32
    )
    weight_map["lm_head.weight"] = shard_file

    # Write safetensors files
    for filename, tensors in shard_tensors.items():
        save_file(tensors, str(tmp_path / filename))

    # Write the index file
    index_data = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map,
    }
    index_path = tmp_path / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    return weight_map


@_requires_safetensors
class TestLoadTensorsFromManifest:
    """Test loading actual tensor data from safetensors files."""

    def test_loads_correct_tensors_for_rank_0(self, tmp_path: Path) -> None:
        """Rank 0 loads embedding and its assigned layer tensors."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert "model.embed_tokens.weight" in loaded
        assert "model.layers.0.self_attn.q_proj.weight" in loaded
        assert "model.layers.1.self_attn.q_proj.weight" not in loaded

    def test_loads_correct_tensors_for_last_rank(self, tmp_path: Path) -> None:
        """Last rank loads norm, lm_head, and its assigned layer tensors."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(3)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        assert "model.norm.weight" in loaded
        assert "lm_head.weight" in loaded
        assert "model.layers.3.self_attn.q_proj.weight" in loaded
        assert "model.embed_tokens.weight" not in loaded

    def test_loaded_tensors_have_correct_dtype(self, tmp_path: Path) -> None:
        """Loaded tensors are cast to the specified dtype."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        for tensor in loaded.values():
            assert tensor.dtype == torch.float32

    def test_loaded_tensors_on_correct_device(self, tmp_path: Path) -> None:
        """Loaded tensors are placed on the specified device."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        for tensor in loaded.values():
            assert tensor.device == torch.device("cpu")

    def test_loads_only_requested_tensors(self, tmp_path: Path) -> None:
        """Only tensors in the manifest are loaded, not the full file."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        # Rank 1 owns only layer 1
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(1)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        # Only layer 1 tensors should be loaded
        for name in loaded:
            assert "model.layers.1." in name
        # Embedding, norm, lm_head, and other layers should NOT be loaded
        assert "model.embed_tokens.weight" not in loaded
        assert "model.norm.weight" not in loaded
        assert "lm_head.weight" not in loaded

    def test_tensor_values_match_source(self, tmp_path: Path) -> None:
        """Loaded tensor values match what was written to the safetensors file."""
        import torch
        from safetensors.torch import save_file

        # Create a simple safetensors file with known values
        known_tensor = torch.ones(4, 4, dtype=torch.float32) * 42.0
        save_file(
            {"model.embed_tokens.weight": known_tensor},
            str(tmp_path / "model-00001-of-00001.safetensors"),
        )

        # Write index
        weight_map = {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
        }
        index_data = {"metadata": {}, "weight_map": weight_map}
        with open(
            tmp_path / "model.safetensors.index.json", "w", encoding="utf-8"
        ) as f:
            json.dump(index_data, f)

        # Create a minimal manifest for rank 0 with just embedding
        manifest = LocalShardManifest(
            tensor_names=["model.embed_tokens.weight"],
            tensor_to_file={
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
            },
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=1,
            local_tensor_count=1,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert torch.allclose(loaded["model.embed_tokens.weight"], known_tensor)

    def test_raises_on_missing_shard_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when a shard file is missing."""
        import torch

        # Create a manifest referencing a non-existent file
        manifest = LocalShardManifest(
            tensor_names=["model.embed_tokens.weight"],
            tensor_to_file={
                "model.embed_tokens.weight": "nonexistent.safetensors"
            },
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=1,
            local_tensor_count=1,
        )

        with pytest.raises(FileNotFoundError, match="Shard file not found"):
            load_tensors_from_manifest(
                manifest=manifest,
                model_path=tmp_path,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )

    def test_raises_on_missing_tensor_in_shard(self, tmp_path: Path) -> None:
        """Raises RuntimeError when a tensor is not found in its shard file."""
        import torch
        from safetensors.torch import save_file

        # Create a shard file with different tensors than expected
        save_file(
            {"some_other_tensor": torch.zeros(4)},
            str(tmp_path / "model-00001-of-00001.safetensors"),
        )

        manifest = LocalShardManifest(
            tensor_names=["model.embed_tokens.weight"],
            tensor_to_file={
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
            },
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=1,
            local_tensor_count=1,
        )

        with pytest.raises(RuntimeError, match="not found in shard file"):
            load_tensors_from_manifest(
                manifest=manifest,
                model_path=tmp_path,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )

    def test_default_dtype_is_bfloat16(self, tmp_path: Path) -> None:
        """When dtype is not specified, tensors default to bfloat16."""
        import torch
        from safetensors.torch import save_file

        save_file(
            {"model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.float32)},
            str(tmp_path / "shard.safetensors"),
        )

        manifest = LocalShardManifest(
            tensor_names=["model.embed_tokens.weight"],
            tensor_to_file={"model.embed_tokens.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=1,
            local_tensor_count=1,
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
        )

        assert loaded["model.embed_tokens.weight"].dtype == torch.bfloat16


@_requires_safetensors
class TestValidateLoadedTensors:
    """Test validation of loaded tensors against manifest."""

    def test_passes_when_all_tensors_present(self, tmp_path: Path) -> None:
        """Validation passes when all manifest tensors are loaded."""
        import torch

        _create_synthetic_safetensors_files(tmp_path, num_layers=4)
        distribution = PipelineLayerDistribution(
            layers_per_rank=(1, 1, 1, 1),
            total_layer_count=4,
            rank_count=4,
        )
        assignment = distribution.get_stage_assignment(0)
        manifest = load_qwen_local_shard_from_safetensors(
            model_path=tmp_path, stage_assignment=assignment
        )

        loaded = load_tensors_from_manifest(
            manifest=manifest,
            model_path=tmp_path,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        # Should not raise
        validate_loaded_tensors(manifest, loaded)

    def test_raises_on_missing_tensors(self) -> None:
        """Validation raises ValueError when tensors are missing."""
        import torch

        manifest = LocalShardManifest(
            tensor_names=[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
            tensor_to_file={
                "model.embed_tokens.weight": "shard.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "shard.safetensors",
            },
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=2,
            local_tensor_count=2,
        )

        # Only provide one of the two expected tensors
        loaded = {
            "model.embed_tokens.weight": torch.zeros(4, 4),
        }

        with pytest.raises(ValueError, match="Missing 1 tensors"):
            validate_loaded_tensors(manifest, loaded)

    def test_raises_with_missing_tensor_names(self) -> None:
        """Error message includes the names of missing tensors."""
        import torch

        manifest = LocalShardManifest(
            tensor_names=[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
            ],
            tensor_to_file={
                "model.embed_tokens.weight": "shard.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "shard.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "shard.safetensors",
            },
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=3,
            local_tensor_count=3,
        )

        # Provide only the embedding
        loaded: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.zeros(4, 4),
        }

        with pytest.raises(ValueError, match="model.layers.0"):
            validate_loaded_tensors(manifest, loaded)

    def test_passes_with_extra_tensors(self) -> None:
        """Validation passes even if loaded dict has extra tensors."""
        import torch

        manifest = LocalShardManifest(
            tensor_names=["model.embed_tokens.weight"],
            tensor_to_file={"model.embed_tokens.weight": "shard.safetensors"},
            stage_assignment=PipelineStageAssignment(
                rank=0,
                start_layer=0,
                end_layer=1,
                owns_embedding=True,
                owns_lm_head=False,
            ),
            total_tensor_count=1,
            local_tensor_count=1,
        )

        # Provide the expected tensor plus an extra one
        loaded = {
            "model.embed_tokens.weight": torch.zeros(4, 4),
            "extra_tensor": torch.zeros(2, 2),
        }

        # Should not raise
        validate_loaded_tensors(manifest, loaded)
