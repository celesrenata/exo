"""
Unit tests for pipeline stage balancing per-layer timing export.

Tests export_per_layer_timing and export_per_stage_timing functions
that extract and aggregate timing data from the PerformanceRecorder.

**Validates: Requirements 6.4**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_INSTRUMENTATION_PATH = _ENGINE_DIR / "instrumentation.py"
_PIPELINE_CONFIG_PATH = _ENGINE_DIR / "pipeline_config.py"
_STAGE_BALANCING_PATH = _ENGINE_DIR / "stage_balancing.py"


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


_instrumentation_mod = _load_module(
    "instrumentation_stage_balancing_isolated", _INSTRUMENTATION_PATH
)
_config_mod = _load_module(
    "pipeline_config_stage_balancing_isolated", _PIPELINE_CONFIG_PATH
)

# Patch sys.modules so stage_balancing.py can import from these modules
sys.modules["exo.worker.engines.pytorch_xpu.instrumentation"] = (
    _instrumentation_mod
)
sys.modules["exo.worker.engines.pytorch_xpu.pipeline_config"] = _config_mod

_stage_balancing_mod = _load_module(
    "stage_balancing_isolated", _STAGE_BALANCING_PATH
)

PerformanceRecorder = _instrumentation_mod.PerformanceRecorder
PipelineLayerDistribution = _config_mod.PipelineLayerDistribution
PerLayerTimingData = _stage_balancing_mod.PerLayerTimingData
PerStageTimingSummary = _stage_balancing_mod.PerStageTimingSummary
export_per_layer_timing = _stage_balancing_mod.export_per_layer_timing
export_per_stage_timing = _stage_balancing_mod.export_per_stage_timing


# ===========================================================================
# Helper to record synthetic layer_compute events
# ===========================================================================


def _record_layer_event(
    recorder: PerformanceRecorder,
    layer_index: int,
    layer_type: str,
    mode: str,
    duration: float,
) -> None:
    """Record a synthetic layer_compute event with controlled duration.

    Uses time.sleep to approximate the desired duration. For test precision,
    we directly inject events instead.
    """
    # We inject events directly into the recorder's internal list for
    # precise control over duration values in tests.
    from exo.worker.engines.pytorch_xpu.instrumentation import (
        PerformanceEvent,
    )

    event = PerformanceEvent(
        event_name="layer_compute",
        start_time=0.0,
        end_time=duration,
        duration=duration,
        mode=mode,  # pyright: ignore[reportArgumentType]
        rank=recorder.rank,
        stage=recorder.stage,
        metadata={
            "layer_index": layer_index,
            "layer_type": layer_type,
        },
    )
    recorder._events.append(event)  # pyright: ignore[reportPrivateUsage]


# ===========================================================================
# Tests for export_per_layer_timing
# ===========================================================================


class TestExportPerLayerTiming:
    """Test export_per_layer_timing extracts and groups events correctly."""

    def test_empty_recorder_returns_empty_list(self) -> None:
        """An empty recorder produces an empty list."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        result = export_per_layer_timing(recorder)
        assert result == []

    def test_single_decode_event(self) -> None:
        """A single decode event produces correct timing data."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        _record_layer_event(recorder, 0, "full_attention", "decode", 0.005)

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        assert result[0].layer_index == 0
        assert result[0].layer_type == "full_attention"
        assert result[0].decode_mean_seconds == pytest.approx(0.005)
        assert result[0].decode_count == 1
        assert result[0].prefill_mean_seconds == 0.0
        assert result[0].prefill_count == 0

    def test_single_prefill_event(self) -> None:
        """A single prefill event produces correct timing data."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        _record_layer_event(
            recorder, 3, "linear_attention", "prefill", 0.010
        )

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        assert result[0].layer_index == 3
        assert result[0].layer_type == "linear_attention"
        assert result[0].decode_mean_seconds == 0.0
        assert result[0].decode_count == 0
        assert result[0].prefill_mean_seconds == pytest.approx(0.010)
        assert result[0].prefill_count == 1

    def test_groups_by_layer_index(self) -> None:
        """Events from different layers are grouped separately."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        _record_layer_event(recorder, 0, "full_attention", "decode", 0.004)
        _record_layer_event(
            recorder, 1, "linear_attention", "decode", 0.006
        )
        _record_layer_event(recorder, 2, "full_attention", "decode", 0.005)

        result = export_per_layer_timing(recorder)
        assert len(result) == 3
        assert result[0].layer_index == 0
        assert result[1].layer_index == 1
        assert result[2].layer_index == 2

    def test_separates_decode_and_prefill(self) -> None:
        """Decode and prefill events for the same layer are separated."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        _record_layer_event(recorder, 5, "full_attention", "decode", 0.003)
        _record_layer_event(recorder, 5, "full_attention", "decode", 0.005)
        _record_layer_event(recorder, 5, "full_attention", "prefill", 0.020)
        _record_layer_event(recorder, 5, "full_attention", "prefill", 0.030)

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        entry = result[0]
        assert entry.layer_index == 5
        assert entry.decode_mean_seconds == pytest.approx(0.004)
        assert entry.decode_count == 2
        assert entry.prefill_mean_seconds == pytest.approx(0.025)
        assert entry.prefill_count == 2

    def test_mean_computation_multiple_events(self) -> None:
        """Mean is computed correctly across multiple events."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        durations = [0.001, 0.002, 0.003, 0.004, 0.005]
        for d in durations:
            _record_layer_event(recorder, 10, "linear_attention", "decode", d)

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        expected_mean = sum(durations) / len(durations)
        assert result[0].decode_mean_seconds == pytest.approx(expected_mean)
        assert result[0].decode_count == 5

    def test_sorted_by_layer_index(self) -> None:
        """Results are sorted by layer_index regardless of insertion order."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        _record_layer_event(recorder, 5, "full_attention", "decode", 0.005)
        _record_layer_event(recorder, 2, "linear_attention", "decode", 0.003)
        _record_layer_event(recorder, 8, "full_attention", "decode", 0.008)
        _record_layer_event(recorder, 0, "linear_attention", "decode", 0.001)

        result = export_per_layer_timing(recorder)
        indices = [entry.layer_index for entry in result]
        assert indices == [0, 2, 5, 8]

    def test_ignores_non_layer_compute_events(self) -> None:
        """Events with names other than 'layer_compute' are ignored."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        # Record a non-layer_compute event via the span API
        with recorder.span(
            "prefill_forward", mode="prefill", metadata={"layer_index": 0}
        ):
            pass
        _record_layer_event(recorder, 0, "full_attention", "decode", 0.005)

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        assert result[0].decode_count == 1

    def test_ignores_events_without_layer_index(self) -> None:
        """layer_compute events without layer_index metadata are ignored."""
        from exo.worker.engines.pytorch_xpu.instrumentation import (
            PerformanceEvent,
        )

        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        # Event with no layer_index in metadata
        event = PerformanceEvent(
            event_name="layer_compute",
            start_time=0.0,
            end_time=0.005,
            duration=0.005,
            mode="decode",
            rank=0,
            stage=0,
            metadata={"layer_type": "full_attention"},
        )
        recorder._events.append(event)  # pyright: ignore[reportPrivateUsage]

        result = export_per_layer_timing(recorder)
        assert result == []

    def test_disabled_recorder_returns_empty(self) -> None:
        """A disabled recorder has no events, returns empty list."""
        recorder = PerformanceRecorder(enabled=False, rank=0, stage=0)
        # Spans on disabled recorder are no-ops
        with recorder.span(
            "layer_compute",
            mode="decode",
            metadata={"layer_index": 0, "layer_type": "full_attention"},
        ):
            pass

        result = export_per_layer_timing(recorder)
        assert result == []

    def test_layer_type_defaults_to_unknown(self) -> None:
        """If layer_type is missing from metadata, defaults to 'unknown'."""
        from exo.worker.engines.pytorch_xpu.instrumentation import (
            PerformanceEvent,
        )

        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)
        event = PerformanceEvent(
            event_name="layer_compute",
            start_time=0.0,
            end_time=0.005,
            duration=0.005,
            mode="decode",
            rank=0,
            stage=0,
            metadata={"layer_index": 7},
        )
        recorder._events.append(event)  # pyright: ignore[reportPrivateUsage]

        result = export_per_layer_timing(recorder)
        assert len(result) == 1
        assert result[0].layer_type == "unknown"


# ===========================================================================
# Tests for export_per_stage_timing
# ===========================================================================


class TestExportPerStageTiming:
    """Test export_per_stage_timing aggregates per-layer data correctly."""

    def test_empty_per_layer_returns_zero_stages(self) -> None:
        """Empty per-layer data produces stages with zero timing."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(4, 4, 4, 4),
            total_layer_count=16,
            rank_count=4,
        )
        result = export_per_stage_timing([], distribution)
        assert len(result) == 4
        for stage in result:
            assert stage.decode_total_mean_seconds == 0.0
            assert stage.prefill_total_mean_seconds == 0.0

    def test_uniform_distribution_aggregation(self) -> None:
        """Aggregates timing correctly for uniform distribution."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2),
            total_layer_count=4,
            rank_count=2,
        )
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.001,
                decode_count=10,
                prefill_mean_seconds=0.010,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=1,
                layer_type="linear_attention",
                decode_mean_seconds=0.002,
                decode_count=10,
                prefill_mean_seconds=0.020,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=2,
                layer_type="full_attention",
                decode_mean_seconds=0.003,
                decode_count=10,
                prefill_mean_seconds=0.030,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=3,
                layer_type="linear_attention",
                decode_mean_seconds=0.004,
                decode_count=10,
                prefill_mean_seconds=0.040,
                prefill_count=1,
            ),
        ]

        result = export_per_stage_timing(per_layer, distribution)
        assert len(result) == 2

        # Rank 0: layers 0-1
        assert result[0].rank == 0
        assert result[0].start_layer == 0
        assert result[0].end_layer == 2
        assert result[0].decode_total_mean_seconds == pytest.approx(0.003)
        assert result[0].prefill_total_mean_seconds == pytest.approx(0.030)
        assert result[0].layer_count == 2

        # Rank 1: layers 2-3
        assert result[1].rank == 1
        assert result[1].start_layer == 2
        assert result[1].end_layer == 4
        assert result[1].decode_total_mean_seconds == pytest.approx(0.007)
        assert result[1].prefill_total_mean_seconds == pytest.approx(0.070)
        assert result[1].layer_count == 2

    def test_nonuniform_distribution(self) -> None:
        """Aggregates correctly for non-uniform distribution."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(3, 1),
            total_layer_count=4,
            rank_count=2,
        )
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.001,
                decode_count=5,
                prefill_mean_seconds=0.010,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=1,
                layer_type="linear_attention",
                decode_mean_seconds=0.002,
                decode_count=5,
                prefill_mean_seconds=0.020,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=2,
                layer_type="full_attention",
                decode_mean_seconds=0.003,
                decode_count=5,
                prefill_mean_seconds=0.030,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=3,
                layer_type="linear_attention",
                decode_mean_seconds=0.004,
                decode_count=5,
                prefill_mean_seconds=0.040,
                prefill_count=1,
            ),
        ]

        result = export_per_stage_timing(per_layer, distribution)
        assert len(result) == 2

        # Rank 0: layers 0, 1, 2
        assert result[0].rank == 0
        assert result[0].start_layer == 0
        assert result[0].end_layer == 3
        assert result[0].decode_total_mean_seconds == pytest.approx(0.006)
        assert result[0].prefill_total_mean_seconds == pytest.approx(0.060)
        assert result[0].layer_count == 3

        # Rank 1: layer 3 only
        assert result[1].rank == 1
        assert result[1].start_layer == 3
        assert result[1].end_layer == 4
        assert result[1].decode_total_mean_seconds == pytest.approx(0.004)
        assert result[1].prefill_total_mean_seconds == pytest.approx(0.040)
        assert result[1].layer_count == 1

    def test_missing_layers_treated_as_zero(self) -> None:
        """Layers without timing data contribute zero to stage totals."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2),
            total_layer_count=4,
            rank_count=2,
        )
        # Only provide timing for layers 0 and 2
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=1,
            ),
            PerLayerTimingData(
                layer_index=2,
                layer_type="full_attention",
                decode_mean_seconds=0.007,
                decode_count=10,
                prefill_mean_seconds=0.070,
                prefill_count=1,
            ),
        ]

        result = export_per_stage_timing(per_layer, distribution)

        # Rank 0: layer 0 has data, layer 1 missing → only layer 0 contributes
        assert result[0].decode_total_mean_seconds == pytest.approx(0.005)
        assert result[0].prefill_total_mean_seconds == pytest.approx(0.050)

        # Rank 1: layer 2 has data, layer 3 missing → only layer 2 contributes
        assert result[1].decode_total_mean_seconds == pytest.approx(0.007)
        assert result[1].prefill_total_mean_seconds == pytest.approx(0.070)

    def test_four_rank_distribution(self) -> None:
        """Works correctly with a 4-rank distribution (production-like)."""
        distribution = PipelineLayerDistribution(
            layers_per_rank=(4, 4, 4, 4),
            total_layer_count=16,
            rank_count=4,
        )
        # Create timing for all 16 layers with increasing decode time
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention" if i % 2 == 0 else "linear_attention",
                decode_mean_seconds=0.001 * (i + 1),
                decode_count=20,
                prefill_mean_seconds=0.010 * (i + 1),
                prefill_count=2,
            )
            for i in range(16)
        ]

        result = export_per_stage_timing(per_layer, distribution)
        assert len(result) == 4

        # Rank 0: layers 0-3, decode means: 0.001, 0.002, 0.003, 0.004
        assert result[0].rank == 0
        assert result[0].start_layer == 0
        assert result[0].end_layer == 4
        assert result[0].decode_total_mean_seconds == pytest.approx(0.010)
        assert result[0].layer_count == 4

        # Rank 3: layers 12-15, decode means: 0.013, 0.014, 0.015, 0.016
        assert result[3].rank == 3
        assert result[3].start_layer == 12
        assert result[3].end_layer == 16
        assert result[3].decode_total_mean_seconds == pytest.approx(0.058)
        assert result[3].layer_count == 4

    def test_end_to_end_with_recorder(self) -> None:
        """Integration: record events, export per-layer, then per-stage."""
        recorder = PerformanceRecorder(enabled=True, rank=0, stage=0)

        # Simulate 4 layers, 3 decode steps each
        for step in range(3):
            for layer_idx in range(4):
                duration = 0.001 * (layer_idx + 1) + 0.0001 * step
                _record_layer_event(
                    recorder,
                    layer_idx,
                    "full_attention" if layer_idx % 2 == 0 else "linear_attention",
                    "decode",
                    duration,
                )

        # Also add prefill events
        for layer_idx in range(4):
            _record_layer_event(
                recorder,
                layer_idx,
                "full_attention" if layer_idx % 2 == 0 else "linear_attention",
                "prefill",
                0.010 * (layer_idx + 1),
            )

        per_layer = export_per_layer_timing(recorder)
        assert len(per_layer) == 4

        # Verify decode counts
        for entry in per_layer:
            assert entry.decode_count == 3
            assert entry.prefill_count == 1

        # Now aggregate into stages
        distribution = PipelineLayerDistribution(
            layers_per_rank=(2, 2),
            total_layer_count=4,
            rank_count=2,
        )
        per_stage = export_per_stage_timing(per_layer, distribution)
        assert len(per_stage) == 2
        assert per_stage[0].layer_count == 2
        assert per_stage[1].layer_count == 2

        # Stage 0 decode total = mean(layer0) + mean(layer1)
        # Stage 1 decode total = mean(layer2) + mean(layer3)
        assert per_stage[0].decode_total_mean_seconds < per_stage[1].decode_total_mean_seconds



# ===========================================================================
# Tests for recommend_pipeline_layer_distribution
# ===========================================================================

# Import the recommendation function and constants
recommend_pipeline_layer_distribution = (
    _stage_balancing_mod.recommend_pipeline_layer_distribution
)
DEFAULT_FIXED_RANK_COSTS = _stage_balancing_mod.DEFAULT_FIXED_RANK_COSTS


class TestRecommendPipelineLayerDistribution:
    """Test recommend_pipeline_layer_distribution produces optimal distributions."""

    def test_uniform_timing_produces_balanced_distribution(self) -> None:
        """When all layers have equal timing, distribution is balanced."""
        num_layers = 16
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=2,
            )
            for i in range(num_layers)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
            fixed_rank_costs={},  # No fixed costs for this test
        )

        assert result.rank_count == 4
        assert result.total_layer_count == 16
        assert sum(result.layers_per_rank) == 16
        # With uniform timing and no fixed costs, each rank gets 4 layers
        assert result.layers_per_rank == (4, 4, 4, 4)

    def test_nonuniform_timing_reduces_max_stage_time(self) -> None:
        """Non-uniform timing produces a distribution that reduces the bottleneck."""
        # Layers 0-7 are fast (0.001s), layers 8-15 are slow (0.003s)
        per_layer = []
        for i in range(16):
            cost = 0.001 if i < 8 else 0.003
            per_layer.append(
                PerLayerTimingData(
                    layer_index=i,
                    layer_type="full_attention",
                    decode_mean_seconds=cost,
                    decode_count=10,
                    prefill_mean_seconds=cost * 10,
                    prefill_count=2,
                )
            )

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
            fixed_rank_costs={},
        )

        assert result.rank_count == 4
        assert result.total_layer_count == 16
        assert sum(result.layers_per_rank) == 16

        # The recommendation should give more layers to the fast section
        # and fewer to the slow section
        # Fast layers (0-7): 0.001 each, slow layers (8-15): 0.003 each
        # Optimal: more fast layers per rank, fewer slow layers per rank
        # Verify the max stage time is less than uniform (4,4,4,4)
        # Uniform max stage: max(4*0.001, 4*0.001, 4*0.003, 4*0.003) = 0.012
        # Recommended should have max stage < 0.012
        max_stage_time = 0.0
        layer_idx = 0
        for rank in range(result.rank_count):
            stage_cost = 0.0
            for _ in range(result.layers_per_rank[rank]):
                stage_cost += per_layer[layer_idx].decode_mean_seconds
                layer_idx += 1
            max_stage_time = max(max_stage_time, stage_cost)

        uniform_max = 4 * 0.003  # 0.012
        assert max_stage_time < uniform_max

    def test_fixed_rank_costs_shift_layers_away(self) -> None:
        """Fixed rank costs cause fewer layers to be assigned to expensive ranks."""
        num_layers = 12
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.002,
                decode_count=10,
                prefill_mean_seconds=0.020,
                prefill_count=2,
            )
            for i in range(num_layers)
        ]

        # No fixed costs: should be balanced (4, 4, 4)
        result_no_fixed = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=3,
            fixed_rank_costs={},
        )

        # With heavy fixed cost on last rank: last rank should get fewer layers
        result_with_fixed = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=3,
            fixed_rank_costs={-1: 0.005},  # 5ms fixed cost on last rank
        )

        assert result_no_fixed.layers_per_rank == (4, 4, 4)
        # Last rank should have fewer layers due to fixed cost
        assert result_with_fixed.layers_per_rank[-1] < result_no_fixed.layers_per_rank[-1]

    def test_result_is_valid_pipeline_layer_distribution(self) -> None:
        """Result is always a valid PipelineLayerDistribution."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="linear_attention",
                decode_mean_seconds=0.001 * (i + 1),
                decode_count=5,
                prefill_mean_seconds=0.010 * (i + 1),
                prefill_count=1,
            )
            for i in range(8)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
        )

        # Validate it's a proper PipelineLayerDistribution
        assert isinstance(result, PipelineLayerDistribution)
        assert result.rank_count == 4
        assert result.total_layer_count == 8
        assert sum(result.layers_per_rank) == 8
        assert len(result.layers_per_rank) == 4
        # Validate contiguous ranges
        assert result.validate_contiguous()

    def test_each_rank_gets_at_least_one_layer(self) -> None:
        """Every rank must have at least 1 layer in the result."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.001 * (i + 1),
                decode_count=10,
                prefill_mean_seconds=0.010,
                prefill_count=1,
            )
            for i in range(8)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
        )

        for count in result.layers_per_rank:
            assert count >= 1

    def test_world_size_one_returns_all_layers(self) -> None:
        """world_size == 1 returns all layers to a single rank."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=2,
            )
            for i in range(64)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=1,
        )

        assert result.rank_count == 1
        assert result.total_layer_count == 64
        assert result.layers_per_rank == (64,)

    def test_prefill_mode(self) -> None:
        """Mode 'prefill' uses prefill timing instead of decode timing."""
        # Decode is uniform, prefill is non-uniform
        per_layer = []
        for i in range(8):
            per_layer.append(
                PerLayerTimingData(
                    layer_index=i,
                    layer_type="full_attention",
                    decode_mean_seconds=0.005,
                    decode_count=10,
                    prefill_mean_seconds=0.001 if i < 4 else 0.010,
                    prefill_count=2,
                )
            )

        result_decode = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=2,
            fixed_rank_costs={},
            mode="decode",
        )

        result_prefill = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=2,
            fixed_rank_costs={},
            mode="prefill",
        )

        # Decode is uniform → balanced (4, 4)
        assert result_decode.layers_per_rank == (4, 4)
        # Prefill is non-uniform → more fast layers in first rank
        assert result_prefill.layers_per_rank[0] > result_prefill.layers_per_rank[1]

    def test_invalid_world_size_raises(self) -> None:
        """world_size < 1 raises ValueError."""
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=2,
            )
        ]

        with pytest.raises(ValueError, match="world_size must be at least 1"):
            recommend_pipeline_layer_distribution(
                per_layer_timing=per_layer,
                world_size=0,
            )

    def test_empty_timing_raises(self) -> None:
        """Empty per_layer_timing raises ValueError."""
        with pytest.raises(ValueError, match="per_layer_timing must not be empty"):
            recommend_pipeline_layer_distribution(
                per_layer_timing=[],
                world_size=4,
            )

    def test_world_size_exceeds_layers_raises(self) -> None:
        """world_size > number of layers raises ValueError."""
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=2,
            )
        ]

        with pytest.raises(ValueError, match="cannot exceed"):
            recommend_pipeline_layer_distribution(
                per_layer_timing=per_layer,
                world_size=2,
            )

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        per_layer = [
            PerLayerTimingData(
                layer_index=0,
                layer_type="full_attention",
                decode_mean_seconds=0.005,
                decode_count=10,
                prefill_mean_seconds=0.050,
                prefill_count=2,
            )
        ]

        with pytest.raises(ValueError, match="mode must be"):
            recommend_pipeline_layer_distribution(
                per_layer_timing=per_layer,
                world_size=1,
                mode="invalid",
            )

    def test_preserves_layer_order(self) -> None:
        """Layer order is preserved — only contiguous ranges are assigned."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.001 * ((i % 3) + 1),
                decode_count=10,
                prefill_mean_seconds=0.010,
                prefill_count=1,
            )
            for i in range(12)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=3,
            fixed_rank_costs={},
        )

        # Verify contiguous assignment
        layer_idx = 0
        for rank in range(result.rank_count):
            assignment = result.get_stage_assignment(rank)
            assert assignment.start_layer == layer_idx
            layer_idx = assignment.end_layer
        assert layer_idx == 12

    def test_default_fixed_rank_costs_used_when_none(self) -> None:
        """When fixed_rank_costs is None, DEFAULT_FIXED_RANK_COSTS is used."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention",
                decode_mean_seconds=0.002,
                decode_count=10,
                prefill_mean_seconds=0.020,
                prefill_count=2,
            )
            for i in range(16)
        ]

        # With default costs (rank 0: 0.0001, last rank: 0.002)
        result_default = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
            fixed_rank_costs=None,
        )

        # With no fixed costs
        result_no_fixed = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
            fixed_rank_costs={},
        )

        # Default costs should cause last rank to get fewer layers
        assert result_default.layers_per_rank[-1] <= result_no_fixed.layers_per_rank[-1]

    def test_large_layer_count(self) -> None:
        """Works correctly with 64 layers (production Qwen3.5-27B)."""
        per_layer = [
            PerLayerTimingData(
                layer_index=i,
                layer_type="full_attention" if i % 4 == 0 else "linear_attention",
                decode_mean_seconds=0.003 if i % 4 == 0 else 0.002,
                decode_count=50,
                prefill_mean_seconds=0.030 if i % 4 == 0 else 0.020,
                prefill_count=5,
            )
            for i in range(64)
        ]

        result = recommend_pipeline_layer_distribution(
            per_layer_timing=per_layer,
            world_size=4,
        )

        assert result.rank_count == 4
        assert result.total_layer_count == 64
        assert sum(result.layers_per_rank) == 64
        for count in result.layers_per_rank:
            assert count >= 1
        assert result.validate_contiguous()
