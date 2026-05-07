"""Property-based tests for pipeline-stage-aware partial model loading.

Uses Hypothesis to verify that filter_state_dict_for_stage() correctly
partitions a model's state_dict according to pipeline stage assignments.

**Validates: Requirements 10.4**
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.model.shard_loader import (
    _extract_layer_index,
    filter_state_dict_for_stage,
)
from exo.worker.engines.pytorch.pipeline.stage import compute_stage_assignments


# --- Strategies ---

# world_size sampled from the supported set {2, 3, 4}
world_size_st = st.sampled_from([2, 3, 4])

# Strategy that generates (total_layers, world_size) pairs where
# total_layers >= world_size and total_layers >= 4
valid_inputs_st = world_size_st.flatmap(
    lambda ws: st.tuples(
        st.integers(min_value=max(ws, 4), max_value=50),
        st.just(ws),
    )
)


def _make_synthetic_state_dict(total_layers: int) -> dict[str, torch.Tensor]:
    """Create a synthetic state_dict with the model.layers.{i}.weight pattern.

    Includes:
      - model.embed_tokens.weight (embedding)
      - model.layers.{i}.weight for each layer i in [0, total_layers)
      - model.norm.weight (final layer norm)
      - lm_head.weight (language model head)
    """
    state: dict[str, torch.Tensor] = {}

    # Embedding
    state["model.embed_tokens.weight"] = torch.zeros(1)

    # Transformer layers
    for i in range(total_layers):
        state[f"model.layers.{i}.weight"] = torch.zeros(1)

    # Final layer norm
    state["model.norm.weight"] = torch.zeros(1)

    # LM head
    state["lm_head.weight"] = torch.zeros(1)

    return state


class TestPartialModelLoadingRespectsStageAssignment:
    """Property 8: Partial model loading respects stage assignment.

    *For any* valid stage assignment (start_layer, end_layer) and any model
    with total_layers layers, the shard loader SHALL load exactly the layers
    in [start_layer, end_layer) and no others. The number of loaded layer
    modules SHALL equal end_layer - start_layer.

    **Validates: Requirements 10.4**
    """

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_each_stage_includes_only_layers_in_range(
        self, inputs: tuple[int, int]
    ) -> None:
        """For each stage, filter_state_dict_for_stage() includes only layers
        in [start_layer, end_layer).

        Requirement 10.4: WHEN loading a model for Pipeline_Parallelism, THE
        Unified_Engine SHALL load only the layers assigned to the local
        Pipeline_Stage.
        """
        total_layers, world_size = inputs
        state_dict = _make_synthetic_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size)

        for assignment in assignments:
            filtered = filter_state_dict_for_stage(state_dict, assignment)

            # Check every key in the filtered dict
            for key in filtered:
                layer_idx = _extract_layer_index(key)
                if layer_idx is not None:
                    assert assignment.start_layer <= layer_idx < assignment.end_layer, (
                        f"Stage rank={assignment.rank} "
                        f"[{assignment.start_layer}, {assignment.end_layer}) "
                        f"should not contain layer {layer_idx} (key={key})"
                    )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_loaded_layer_count_equals_stage_size(
        self, inputs: tuple[int, int]
    ) -> None:
        """The number of unique layer indices in the filtered dict equals
        end_layer - start_layer.

        Requirement 10.4: The number of loaded layer modules SHALL equal
        end_layer - start_layer.
        """
        total_layers, world_size = inputs
        state_dict = _make_synthetic_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size)

        for assignment in assignments:
            filtered = filter_state_dict_for_stage(state_dict, assignment)

            # Count unique layer indices in the filtered dict
            layer_indices: set[int] = set()
            for key in filtered:
                layer_idx = _extract_layer_index(key)
                if layer_idx is not None:
                    layer_indices.add(layer_idx)

            expected_count = assignment.end_layer - assignment.start_layer
            assert len(layer_indices) == expected_count, (
                f"Stage rank={assignment.rank} "
                f"[{assignment.start_layer}, {assignment.end_layer}): "
                f"expected {expected_count} layers, got {len(layer_indices)} "
                f"(indices={sorted(layer_indices)})"
            )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_no_layer_outside_range_appears(
        self, inputs: tuple[int, int]
    ) -> None:
        """No layer outside the assigned range appears in the filtered dict.

        Requirement 10.4: SHALL load only the layers assigned to the local
        Pipeline_Stage — no others.
        """
        total_layers, world_size = inputs
        state_dict = _make_synthetic_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size)

        for assignment in assignments:
            filtered = filter_state_dict_for_stage(state_dict, assignment)

            for key in filtered:
                layer_idx = _extract_layer_index(key)
                if layer_idx is not None:
                    # Must NOT be outside the range
                    assert layer_idx >= assignment.start_layer, (
                        f"Layer {layer_idx} is below start_layer "
                        f"{assignment.start_layer} for rank {assignment.rank}"
                    )
                    assert layer_idx < assignment.end_layer, (
                        f"Layer {layer_idx} is at or above end_layer "
                        f"{assignment.end_layer} for rank {assignment.rank}"
                    )

    @settings(max_examples=100)
    @given(inputs=valid_inputs_st)
    def test_all_layers_appear_exactly_once_across_stages(
        self, inputs: tuple[int, int]
    ) -> None:
        """Across all stages, every layer appears exactly once (no gaps, no
        overlaps).

        Requirement 10.4: The full model is partitioned across stages with
        each layer assigned to exactly one stage.
        """
        total_layers, world_size = inputs
        state_dict = _make_synthetic_state_dict(total_layers)
        assignments = compute_stage_assignments(total_layers, world_size)

        # Collect all layer indices across all stages
        all_layer_indices: list[int] = []

        for assignment in assignments:
            filtered = filter_state_dict_for_stage(state_dict, assignment)

            for key in filtered:
                layer_idx = _extract_layer_index(key)
                if layer_idx is not None:
                    all_layer_indices.append(layer_idx)

        # Every layer from 0 to total_layers-1 should appear exactly once
        expected_layers = set(range(total_layers))
        actual_layers = set(all_layer_indices)

        # No gaps
        missing = expected_layers - actual_layers
        assert not missing, (
            f"Layers missing from all stages: {sorted(missing)}"
        )

        # No overlaps (duplicates)
        assert len(all_layer_indices) == len(actual_layers), (
            f"Duplicate layers found across stages: "
            f"total entries={len(all_layer_indices)}, "
            f"unique={len(actual_layers)}"
        )

        # Complete coverage
        assert actual_layers == expected_layers
