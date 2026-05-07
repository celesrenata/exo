"""Unit tests for pipeline-stage-aware partial model loading.

Tests filter_state_dict_for_stage, validate_shard_load, and load_shard_for_stage
using synthetic state_dicts that mimic transformer model naming conventions.
"""

from __future__ import annotations

import pytest
import torch

from exo.worker.engines.pytorch.model.shard_loader import (
    ShardLoadResult,
    _extract_layer_index,
    filter_state_dict_for_stage,
    load_shard_for_stage,
    move_state_dict_to_device,
    validate_shard_load,
)
from exo.worker.engines.pytorch.pipeline.stage import (
    StageAssignment,
    compute_stage_assignments,
)


def _make_state_dict(total_layers: int = 8) -> dict[str, torch.Tensor]:
    """Create a synthetic state_dict mimicking a transformer model.

    Includes:
      - model.embed_tokens.weight
      - model.layers.{i}.self_attn.q_proj.weight for each layer
      - model.layers.{i}.self_attn.k_proj.weight for each layer
      - model.layers.{i}.mlp.gate_proj.weight for each layer
      - model.norm.weight
      - lm_head.weight
    """
    state: dict[str, torch.Tensor] = {}

    # Embedding
    state["model.embed_tokens.weight"] = torch.randn(1000, 128)

    # Transformer layers
    for i in range(total_layers):
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(128, 128)
        state[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(128, 128)
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(256, 128)

    # Final layer norm
    state["model.norm.weight"] = torch.randn(128)

    # LM head
    state["lm_head.weight"] = torch.randn(1000, 128)

    return state


class TestExtractLayerIndex:
    """Tests for _extract_layer_index helper."""

    def test_valid_layer_key(self) -> None:
        assert _extract_layer_index("model.layers.0.self_attn.q_proj.weight") == 0
        assert _extract_layer_index("model.layers.7.mlp.gate_proj.weight") == 7
        assert _extract_layer_index("model.layers.27.self_attn.k_proj.weight") == 27

    def test_non_layer_keys(self) -> None:
        assert _extract_layer_index("model.embed_tokens.weight") is None
        assert _extract_layer_index("lm_head.weight") is None
        assert _extract_layer_index("model.norm.weight") is None

    def test_malformed_layer_key(self) -> None:
        assert _extract_layer_index("model.layers.") is None
        assert _extract_layer_index("model.layers.abc.weight") is None


class TestFilterStateDictForStage:
    """Tests for filter_state_dict_for_stage."""

    def test_first_stage_gets_embedding_and_layers(self) -> None:
        state_dict = _make_state_dict(8)
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=4,
            has_embedding=True, has_lm_head=False,
        )

        filtered = filter_state_dict_for_stage(state_dict, assignment)

        # Should have embedding
        assert "model.embed_tokens.weight" in filtered
        # Should have layers 0-3
        assert "model.layers.0.self_attn.q_proj.weight" in filtered
        assert "model.layers.3.mlp.gate_proj.weight" in filtered
        # Should NOT have layers 4+
        assert "model.layers.4.self_attn.q_proj.weight" not in filtered
        # Should NOT have lm_head or norm
        assert "lm_head.weight" not in filtered
        assert "model.norm.weight" not in filtered

    def test_last_stage_gets_lm_head_and_norm(self) -> None:
        state_dict = _make_state_dict(8)
        assignment = StageAssignment(
            rank=1, start_layer=4, end_layer=8,
            has_embedding=False, has_lm_head=True,
        )

        filtered = filter_state_dict_for_stage(state_dict, assignment)

        # Should have lm_head and norm
        assert "lm_head.weight" in filtered
        assert "model.norm.weight" in filtered
        # Should have layers 4-7
        assert "model.layers.4.self_attn.q_proj.weight" in filtered
        assert "model.layers.7.mlp.gate_proj.weight" in filtered
        # Should NOT have layers 0-3
        assert "model.layers.0.self_attn.q_proj.weight" not in filtered
        # Should NOT have embedding
        assert "model.embed_tokens.weight" not in filtered

    def test_middle_stage_gets_only_layers(self) -> None:
        state_dict = _make_state_dict(12)
        assignment = StageAssignment(
            rank=1, start_layer=4, end_layer=8,
            has_embedding=False, has_lm_head=False,
        )

        filtered = filter_state_dict_for_stage(state_dict, assignment)

        # Should have layers 4-7 only
        assert "model.layers.4.self_attn.q_proj.weight" in filtered
        assert "model.layers.7.mlp.gate_proj.weight" in filtered
        # Should NOT have embedding, lm_head, norm, or other layers
        assert "model.embed_tokens.weight" not in filtered
        assert "lm_head.weight" not in filtered
        assert "model.norm.weight" not in filtered
        assert "model.layers.0.self_attn.q_proj.weight" not in filtered
        assert "model.layers.8.self_attn.q_proj.weight" not in filtered

    def test_full_pipeline_covers_all_layers(self) -> None:
        """All layers appear exactly once across all stages."""
        total_layers = 12
        state_dict = _make_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size=4)

        all_keys: set[str] = set()
        for assignment in assignments:
            filtered = filter_state_dict_for_stage(state_dict, assignment)
            # No overlap
            overlap = all_keys & set(filtered.keys())
            assert overlap == set(), f"Overlap found: {overlap}"
            all_keys.update(filtered.keys())

        # All keys covered
        assert all_keys == set(state_dict.keys())

    def test_empty_state_dict(self) -> None:
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=4,
            has_embedding=True, has_lm_head=False,
        )
        filtered = filter_state_dict_for_stage({}, assignment)
        assert filtered == {}


class TestMoveStateDictToDevice:
    """Tests for move_state_dict_to_device."""

    def test_moves_to_cpu(self) -> None:
        """Test moving tensors to CPU (always available)."""
        state_dict = {"a": torch.randn(4, 4), "b": torch.randn(2)}
        moved = move_state_dict_to_device(state_dict, "cpu")
        for key, tensor in moved.items():
            assert tensor.device.type == "cpu"
        assert set(moved.keys()) == {"a", "b"}


class TestValidateShardLoad:
    """Tests for validate_shard_load."""

    def test_valid_result_passes(self) -> None:
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=4,
            has_embedding=True, has_lm_head=False,
        )
        result = ShardLoadResult(
            tensor_count=13,
            layer_count=4,
            has_embedding=True,
            has_lm_head=False,
            has_layer_norm=False,
            device="cpu",
        )
        # Should not raise
        validate_shard_load(result, assignment)

    def test_wrong_layer_count_raises(self) -> None:
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=4,
            has_embedding=True, has_lm_head=False,
        )
        result = ShardLoadResult(
            tensor_count=10,
            layer_count=3,  # Expected 4
            has_embedding=True,
            has_lm_head=False,
            has_layer_norm=False,
            device="cpu",
        )
        with pytest.raises(ValueError, match="expected 4 layers"):
            validate_shard_load(result, assignment)

    def test_missing_embedding_raises(self) -> None:
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=4,
            has_embedding=True, has_lm_head=False,
        )
        result = ShardLoadResult(
            tensor_count=12,
            layer_count=4,
            has_embedding=False,  # Should be True
            has_lm_head=False,
            has_layer_norm=False,
            device="cpu",
        )
        with pytest.raises(ValueError, match="expected embedding tensors"):
            validate_shard_load(result, assignment)

    def test_missing_lm_head_raises(self) -> None:
        assignment = StageAssignment(
            rank=1, start_layer=4, end_layer=8,
            has_embedding=False, has_lm_head=True,
        )
        result = ShardLoadResult(
            tensor_count=12,
            layer_count=4,
            has_embedding=False,
            has_lm_head=False,  # Should be True
            has_layer_norm=False,
            device="cpu",
        )
        with pytest.raises(ValueError, match="expected LM head tensors"):
            validate_shard_load(result, assignment)

    def test_no_embedding_expected_no_embedding_loaded(self) -> None:
        """Middle stage: no embedding expected, none loaded — passes."""
        assignment = StageAssignment(
            rank=1, start_layer=4, end_layer=8,
            has_embedding=False, has_lm_head=False,
        )
        result = ShardLoadResult(
            tensor_count=12,
            layer_count=4,
            has_embedding=False,
            has_lm_head=False,
            has_layer_norm=False,
            device="cpu",
        )
        # Should not raise
        validate_shard_load(result, assignment)


class TestLoadShardForStage:
    """Integration tests for load_shard_for_stage using CPU device."""

    def test_first_stage_load(self) -> None:
        """Load first stage on CPU (simulating GPU load)."""
        state_dict = _make_state_dict(8)
        assignments = compute_stage_assignments(8, world_size=2)
        # First stage: rank 0, layers 0-3, has_embedding
        assignment = assignments[0]

        # Use CPU as target since GPU may not be available in CI
        on_device, result = load_shard_for_stage(
            state_dict, assignment, device_type="cpu", device_index=0  # type: ignore[arg-type]
        )

        assert result.layer_count == 4
        assert result.has_embedding is True
        assert result.has_lm_head is False
        assert result.tensor_count > 0
        assert result.device == "cpu:0"

    def test_last_stage_load(self) -> None:
        """Load last stage on CPU."""
        state_dict = _make_state_dict(8)
        assignments = compute_stage_assignments(8, world_size=2)
        # Last stage: rank 1, layers 4-7, has_lm_head
        assignment = assignments[1]

        on_device, result = load_shard_for_stage(
            state_dict, assignment, device_type="cpu", device_index=0  # type: ignore[arg-type]
        )

        assert result.layer_count == 4
        assert result.has_embedding is False
        assert result.has_lm_head is True
        assert result.has_layer_norm is True
        assert result.tensor_count > 0

    def test_four_stage_pipeline(self) -> None:
        """Load all 4 stages and verify complete coverage."""
        total_layers = 28
        state_dict = _make_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size=4)

        total_tensors = 0
        all_keys: set[str] = set()

        for assignment in assignments:
            on_device, result = load_shard_for_stage(
                state_dict, assignment, device_type="cpu", device_index=0  # type: ignore[arg-type]
            )
            total_tensors += result.tensor_count
            keys = set(on_device.keys())
            # No overlap
            assert all_keys.isdisjoint(keys)
            all_keys.update(keys)

        # All tensors accounted for
        assert all_keys == set(state_dict.keys())
        assert total_tensors == len(state_dict)

    def test_validation_failure_on_missing_layers(self) -> None:
        """If state_dict is missing layers, validation should fail."""
        # Create state_dict with only layers 0-3 but assign layers 0-7
        state_dict = _make_state_dict(4)
        assignment = StageAssignment(
            rank=0, start_layer=0, end_layer=8,
            has_embedding=True, has_lm_head=False,
        )

        with pytest.raises(ValueError, match="expected 8 layers"):
            load_shard_for_stage(
                state_dict, assignment, device_type="cpu", device_index=0  # type: ignore[arg-type]
            )
