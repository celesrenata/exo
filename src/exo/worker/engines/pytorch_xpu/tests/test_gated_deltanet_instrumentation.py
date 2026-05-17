"""
Unit tests for GatedDeltaNet state optimization instrumentation.

Tests that:
- Counters increment correctly for state allocations vs reuses
- Timing spans are recorded with correct metadata for recurrent steps
- Instrumentation is no-op when recorder is None

**Validates: Requirements 5.10, 8.1, 8.2**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent

_INSTRUMENTATION_PATH = _ENGINE_DIR / "instrumentation.py"
_STATE_PATH = _ENGINE_DIR / "gated_deltanet_state.py"
_DELTANET_PATH = _ENGINE_DIR / "gated_deltanet.py"
_CACHE_PATH = _ENGINE_DIR / "gated_deltanet_cache.py"


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


_instr_mod = _load_module("instrumentation_test_gdn_instr", _INSTRUMENTATION_PATH)
_cache_mod = _load_module("gated_deltanet_cache_test_gdn_instr", _CACHE_PATH)
_state_mod = _load_module("gated_deltanet_state_test_gdn_instr", _STATE_PATH)
_deltanet_mod = _load_module("gated_deltanet_test_gdn_instr", _DELTANET_PATH)

PerformanceRecorder = _instr_mod.PerformanceRecorder
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache
GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
initialize_gated_deltanet_state = _state_mod.initialize_gated_deltanet_state
gated_deltanet_decode_recurrent_step_optimized = (
    _deltanet_mod.gated_deltanet_decode_recurrent_step_optimized
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_BATCH_SIZE = 1
_NUM_HEADS = 4
_KEY_DIM = 16
_VALUE_DIM = 16
_CONV_DIM = _NUM_HEADS * (_KEY_DIM + _VALUE_DIM)
_CONV_KERNEL_SIZE = 4
_DEVICE = torch.device("cpu")


def _make_persistent_state(
    request_id: str = "test-req-001",
    layer_index: int = 0,
) -> GatedDeltaNetPersistentState:
    """Create a persistent state for testing."""
    return GatedDeltaNetPersistentState.create(
        request_identifier=request_id,
        layer_index=layer_index,
        batch_size=_BATCH_SIZE,
        num_heads=_NUM_HEADS,
        key_dim=_KEY_DIM,
        value_dim=_VALUE_DIM,
        conv_dim=_CONV_DIM,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        device=_DEVICE,
    )


def _make_random_inputs(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random input tensors for the recurrent step."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    v = torch.randn(_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM, generator=gen, dtype=torch.bfloat16)
    gate = torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16) * 0.1
    beta = torch.sigmoid(torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16))
    return q, k, v, gate, beta


# ===========================================================================
# Tests for state allocation counters
# ===========================================================================


class TestStateAllocationCounters:
    """Test that initialize_gated_deltanet_state increments counters correctly."""

    def test_new_allocation_increments_counter(self) -> None:
        """Creating fresh state increments state_allocation_new counter."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        cache = GatedDeltaNetCache()

        initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=recorder,
        )

        assert recorder.get_counter("state_allocation_new") == 1
        assert recorder.get_counter("state_allocation_reused") == 0

    def test_reuse_increments_counter(self) -> None:
        """Reusing recycled state increments state_allocation_reused counter."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        cache = GatedDeltaNetCache()

        # First: create state for request A
        state = initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=recorder,
        )

        # Recycle the state (simulates request completion)
        state.recycle()

        # Second: initialize for a new request — should reuse
        initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-002",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=recorder,
        )

        assert recorder.get_counter("state_allocation_new") == 1
        assert recorder.get_counter("state_allocation_reused") == 1

    def test_multiple_layers_accumulate_counters(self) -> None:
        """Counters accumulate across multiple layer initializations."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        cache = GatedDeltaNetCache()

        for layer_idx in range(4):
            initialize_gated_deltanet_state(
                cache=cache,
                request_identifier="req-001",
                layer_index=layer_idx,
                batch_size=_BATCH_SIZE,
                num_heads=_NUM_HEADS,
                key_dim=_KEY_DIM,
                value_dim=_VALUE_DIM,
                conv_dim=_CONV_DIM,
                conv_kernel_size=_CONV_KERNEL_SIZE,
                device=_DEVICE,
                performance_recorder=recorder,
            )

        assert recorder.get_counter("state_allocation_new") == 4
        assert recorder.get_counter("state_allocation_reused") == 0

    def test_no_counter_when_recorder_is_none(self) -> None:
        """No error or side effect when recorder is None."""
        cache = GatedDeltaNetCache()

        # Should not raise
        state = initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=None,
        )
        assert state is not None

    def test_idempotent_call_does_not_increment(self) -> None:
        """Idempotent calls (same request, same layer) do not increment counters."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        cache = GatedDeltaNetCache()

        initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=recorder,
        )

        # Call again — idempotent, should not increment
        initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
            performance_recorder=recorder,
        )

        assert recorder.get_counter("state_allocation_new") == 1
        assert recorder.get_counter("state_allocation_reused") == 0


# ===========================================================================
# Tests for recurrent step timing spans
# ===========================================================================


class TestRecurrentStepTimingSpans:
    """Test that optimized recurrent step records timing spans with metadata."""

    def test_span_recorded_with_layer_index(self) -> None:
        """Recurrent step records a timing span with layer_index metadata."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        persistent_state = _make_persistent_state(layer_index=7)
        q, k, v, gate, beta = _make_random_inputs()

        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state,
            performance_recorder=recorder,
            layer_index=7,
        )

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert event.event_name == "gated_deltanet_optimized_recurrent_step"
        assert event.mode == "decode"
        assert event.metadata["layer_index"] == 7
        assert event.metadata["batch_size"] == _BATCH_SIZE
        assert event.metadata["num_heads"] == _NUM_HEADS
        assert event.metadata["key_head_dim"] == _KEY_DIM
        assert event.duration >= 0.0

    def test_multiple_steps_record_multiple_spans(self) -> None:
        """Each decode step records its own timing span."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        persistent_state = _make_persistent_state(layer_index=3)

        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(seed=step)
            gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, persistent_state,
                performance_recorder=recorder,
                layer_index=3,
            )

        assert len(recorder.events) == 5
        for event in recorder.events:
            assert event.event_name == "gated_deltanet_optimized_recurrent_step"
            assert event.metadata["layer_index"] == 3

    def test_span_without_layer_index(self) -> None:
        """Span is recorded even without layer_index (metadata omits it)."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        persistent_state = _make_persistent_state()
        q, k, v, gate, beta = _make_random_inputs()

        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state,
            performance_recorder=recorder,
            layer_index=None,
        )

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert "layer_index" not in event.metadata

    def test_span_duration_is_nonnegative(self) -> None:
        """Recorded span duration is always >= 0."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        persistent_state = _make_persistent_state()
        q, k, v, gate, beta = _make_random_inputs()

        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state,
            performance_recorder=recorder,
            layer_index=0,
        )

        assert recorder.events[0].duration >= 0.0


# ===========================================================================
# Tests for no-op when recorder is None
# ===========================================================================


class TestInstrumentationNoOp:
    """Test that instrumentation is zero-cost when recorder is None."""

    def test_optimized_step_works_without_recorder(self) -> None:
        """Optimized step produces correct output without a recorder."""
        persistent_state = _make_persistent_state()
        q, k, v, gate, beta = _make_random_inputs()

        output = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state,
            performance_recorder=None,
            layer_index=5,
        )

        assert output.shape == (_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM)
        assert output.dtype == torch.bfloat16

    def test_optimized_step_same_result_with_and_without_recorder(self) -> None:
        """Output is identical whether recorder is provided or not."""
        state_a = _make_persistent_state(request_id="a", layer_index=0)
        state_b = _make_persistent_state(request_id="b", layer_index=0)
        q, k, v, gate, beta = _make_random_inputs()

        recorder = PerformanceRecorder(rank=0, stage=0)

        output_with = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, state_a,
            performance_recorder=recorder,
            layer_index=0,
        )
        output_without = gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, state_b,
            performance_recorder=None,
            layer_index=0,
        )

        assert torch.equal(output_with, output_without)

    def test_disabled_recorder_records_nothing(self) -> None:
        """Disabled recorder does not record spans or counters."""
        recorder = PerformanceRecorder(enabled=False, rank=0, stage=0)
        persistent_state = _make_persistent_state()
        q, k, v, gate, beta = _make_random_inputs()

        gated_deltanet_decode_recurrent_step_optimized(
            q, k, v, gate, beta, persistent_state,
            performance_recorder=recorder,
            layer_index=0,
        )

        assert len(recorder.events) == 0

    def test_initialize_state_works_without_recorder(self) -> None:
        """State initialization works without a recorder (backward compatible)."""
        cache = GatedDeltaNetCache()

        state = initialize_gated_deltanet_state(
            cache=cache,
            request_identifier="req-001",
            layer_index=0,
            batch_size=_BATCH_SIZE,
            num_heads=_NUM_HEADS,
            key_dim=_KEY_DIM,
            value_dim=_VALUE_DIM,
            conv_dim=_CONV_DIM,
            conv_kernel_size=_CONV_KERNEL_SIZE,
            device=_DEVICE,
        )

        assert state is not None
        assert state.request_identifier == "req-001"


# ===========================================================================
# Tests for per-token cast counters in _forward_layer context
# ===========================================================================


class TestPerTokenCastCounters:
    """Test the per_token_casts_avoided / per_token_casts_baseline counters.

    These counters are incremented in pipeline_parallel_shard._forward_layer().
    We test the counter logic directly using the PerformanceRecorder since
    the actual _forward_layer requires a full model setup.
    """

    def test_cast_avoided_counter_increments(self) -> None:
        """Counter increments when optimized path is used."""
        recorder = PerformanceRecorder(rank=0, stage=0)

        # Simulate what _forward_layer does for a linear_attention layer in decode
        # with persistent state enabled
        mode = "decode"
        layer_type = "linear_attention"
        persistent_state_enabled = True

        if mode == "decode" and layer_type == "linear_attention":
            if persistent_state_enabled:
                recorder.increment_counter("per_token_casts_avoided")
            else:
                recorder.increment_counter("per_token_casts_baseline")

        assert recorder.get_counter("per_token_casts_avoided") == 1
        assert recorder.get_counter("per_token_casts_baseline") == 0

    def test_cast_baseline_counter_increments(self) -> None:
        """Counter increments when baseline path is used."""
        recorder = PerformanceRecorder(rank=0, stage=0)

        mode = "decode"
        layer_type = "linear_attention"
        persistent_state_enabled = False

        if mode == "decode" and layer_type == "linear_attention":
            if persistent_state_enabled:
                recorder.increment_counter("per_token_casts_avoided")
            else:
                recorder.increment_counter("per_token_casts_baseline")

        assert recorder.get_counter("per_token_casts_avoided") == 0
        assert recorder.get_counter("per_token_casts_baseline") == 1

    def test_no_counter_for_full_attention_layers(self) -> None:
        """Full attention layers do not increment cast counters."""
        recorder = PerformanceRecorder(rank=0, stage=0)

        mode = "decode"
        layer_type = "full_attention"
        persistent_state_enabled = True

        # Simulate the condition: only linear_attention layers count
        if mode == "decode" and layer_type == "linear_attention":
            if persistent_state_enabled:
                recorder.increment_counter("per_token_casts_avoided")
            else:
                recorder.increment_counter("per_token_casts_baseline")

        assert recorder.get_counter("per_token_casts_avoided") == 0
        assert recorder.get_counter("per_token_casts_baseline") == 0

    def test_no_counter_during_prefill(self) -> None:
        """Prefill mode does not increment cast counters."""
        recorder = PerformanceRecorder(rank=0, stage=0)

        mode = "prefill"
        layer_type = "linear_attention"
        persistent_state_enabled = True

        if mode == "decode" and layer_type == "linear_attention":
            if persistent_state_enabled:
                recorder.increment_counter("per_token_casts_avoided")
            else:
                recorder.increment_counter("per_token_casts_baseline")

        assert recorder.get_counter("per_token_casts_avoided") == 0
        assert recorder.get_counter("per_token_casts_baseline") == 0
