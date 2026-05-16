"""
Unit tests for performance instrumentation module.

Tests PerformanceRecorder span duration nonnegative property, JSON serialization
round trip, counter aggregation, and disabled-mode minimal behavior.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.6, 8.8, 8.10**
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_INSTRUMENTATION_PATH = _THIS_DIR.parent / "instrumentation.py"


def _load_instrumentation() -> types.ModuleType:
    """Load instrumentation.py directly from file, avoiding __init__.py."""
    module_name = "instrumentation_unit_isolated"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _INSTRUMENTATION_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_instrumentation()
PerformanceEvent = _mod.PerformanceEvent
PerformanceSummary = _mod.PerformanceSummary
PerformanceRecorder = _mod.PerformanceRecorder
deserialize_events_from_json = _mod.deserialize_events_from_json


# ===========================================================================
# Tests for PerformanceEvent model
# ===========================================================================


class TestPerformanceEvent:
    """Test PerformanceEvent immutability and field constraints."""

    def test_event_is_frozen(self) -> None:
        """PerformanceEvent instances are immutable."""
        event = PerformanceEvent(
            event_name="test_span",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
        )
        with pytest.raises(ValidationError):
            event.event_name = "modified"  # type: ignore[misc]

    def test_event_mode_classification(self) -> None:
        """Events are classified as prefill or decode, never both."""
        prefill_event = PerformanceEvent(
            event_name="forward",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            mode="prefill",
        )
        decode_event = PerformanceEvent(
            event_name="forward",
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            mode="decode",
        )
        assert prefill_event.mode == "prefill"
        assert decode_event.mode == "decode"
        assert prefill_event.mode != decode_event.mode

    def test_event_with_parent_reference(self) -> None:
        """Events can reference a parent event by name."""
        event = PerformanceEvent(
            event_name="layer_compute",
            start_time=1.0,
            end_time=1.5,
            duration=0.5,
            parent_event_name="forward_pass",
        )
        assert event.parent_event_name == "forward_pass"

    def test_event_with_metadata(self) -> None:
        """Events can carry arbitrary metadata."""
        event = PerformanceEvent(
            event_name="communication",
            start_time=0.0,
            end_time=0.1,
            duration=0.1,
            metadata={"bytes_sent": 4096, "destination_rank": 1},
        )
        assert event.metadata["bytes_sent"] == 4096
        assert event.metadata["destination_rank"] == 1


# ===========================================================================
# Tests for PerformanceRecorder span API
# ===========================================================================


class TestPerformanceRecorderSpan:
    """Test context-manager span API timing and nesting.

    **Validates: Requirements 8.1, 8.2, 8.4**
    """

    def test_span_duration_is_nonnegative(self) -> None:
        """All measured durations are >= 0 (Nonnegative Duration Property)."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("fast_operation", mode="decode"):
            pass  # Near-zero duration operation
        assert len(recorder.events) == 1
        assert recorder.events[0].duration >= 0.0

    def test_span_measures_elapsed_time(self) -> None:
        """Span duration reflects actual elapsed time."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        sleep_duration = 0.01
        with recorder.span("timed_operation", mode="prefill"):
            time.sleep(sleep_duration)
        event = recorder.events[0]
        assert event.duration >= sleep_duration * 0.9  # Allow small timing variance
        assert event.end_time >= event.start_time

    def test_nested_spans_have_parent_child_relationship(self) -> None:
        """Parent timing spans contain child timing spans (Timing Nesting Property)."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("parent_operation", mode="decode"):
            time.sleep(0.005)
            with recorder.span("child_operation", mode="decode"):
                time.sleep(0.005)

        events = recorder.events
        assert len(events) == 2

        # Child is recorded first (inner context exits first)
        child_event = events[0]
        parent_event = events[1]

        assert child_event.event_name == "child_operation"
        assert parent_event.event_name == "parent_operation"
        assert child_event.parent_event_name == "parent_operation"

        # Parent duration contains child duration
        assert parent_event.duration >= child_event.duration
        # Parent start <= child start
        assert parent_event.start_time <= child_event.start_time
        # Parent end >= child end
        assert parent_event.end_time >= child_event.end_time

    def test_explicit_parent_overrides_implicit(self) -> None:
        """Explicit parent parameter takes precedence over stack-based parent."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("outer"), recorder.span("inner", parent="explicit_parent"):
            pass

        inner_event = recorder.events[0]
        assert inner_event.parent_event_name == "explicit_parent"

    def test_span_records_mode(self) -> None:
        """Spans correctly record prefill or decode mode."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("prefill_step", mode="prefill"):
            pass
        with recorder.span("decode_step", mode="decode"):
            pass

        assert recorder.events[0].mode == "prefill"
        assert recorder.events[1].mode == "decode"

    def test_span_records_rank_and_stage(self) -> None:
        """Spans inherit rank and stage from the recorder."""
        recorder = PerformanceRecorder(rank=2, stage=3)
        with recorder.span("operation"):
            pass

        event = recorder.events[0]
        assert event.rank == 2
        assert event.stage == 3

    def test_span_with_metadata(self) -> None:
        """Spans can carry metadata."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("send_activation", metadata={"tensor_size": 1024}):
            pass

        event = recorder.events[0]
        assert event.metadata["tensor_size"] == 1024


# ===========================================================================
# Tests for PerformanceRecorder counters
# ===========================================================================


class TestPerformanceRecorderCounters:
    """Test counter aggregation.

    **Validates: Requirements 8.1, 8.2**
    """

    def test_counter_starts_at_zero(self) -> None:
        """Uninitialized counters return 0."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        assert recorder.get_counter("nonexistent") == 0

    def test_counter_increments(self) -> None:
        """Counters accumulate correctly."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        recorder.increment_counter("messages_sent")
        recorder.increment_counter("messages_sent")
        recorder.increment_counter("messages_sent")
        assert recorder.get_counter("messages_sent") == 3

    def test_counter_increment_by_amount(self) -> None:
        """Counters can be incremented by arbitrary amounts."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        recorder.increment_counter("bytes_transferred", amount=4096)
        recorder.increment_counter("bytes_transferred", amount=2048)
        assert recorder.get_counter("bytes_transferred") == 6144

    def test_multiple_counters_independent(self) -> None:
        """Different counters are independent."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        recorder.increment_counter("allocations", amount=5)
        recorder.increment_counter("deallocations", amount=3)
        assert recorder.get_counter("allocations") == 5
        assert recorder.get_counter("deallocations") == 3

    def test_counters_in_summary(self) -> None:
        """Counters appear in the summary."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        recorder.increment_counter("fast_path_messages", amount=10)
        recorder.increment_counter("generic_messages", amount=2)
        summary = recorder.summarize()
        assert summary.counters["fast_path_messages"] == 10
        assert summary.counters["generic_messages"] == 2


# ===========================================================================
# Tests for disabled mode
# ===========================================================================


class TestPerformanceRecorderDisabled:
    """Test that disabled mode has minimal behavior.

    **Validates: Requirements 8.4, 8.10**
    """

    def test_disabled_span_records_nothing(self) -> None:
        """When disabled, spans do not record events."""
        recorder = PerformanceRecorder(enabled=False, rank=0, stage=0)
        with recorder.span("operation", mode="decode"):
            time.sleep(0.001)
        assert len(recorder.events) == 0

    def test_disabled_counter_records_nothing(self) -> None:
        """When disabled, counters do not increment."""
        recorder = PerformanceRecorder(enabled=False, rank=0, stage=0)
        recorder.increment_counter("messages")
        assert recorder.get_counter("messages") == 0

    def test_disabled_summary_is_empty(self) -> None:
        """When disabled, summary shows zero events."""
        recorder = PerformanceRecorder(enabled=False, rank=0, stage=0)
        with recorder.span("operation"):
            pass
        summary = recorder.summarize()
        assert summary.total_event_count == 0


# ===========================================================================
# Tests for JSON serialization round trip
# ===========================================================================


class TestJsonSerialization:
    """Test JSON serialization preserves all numeric fields.

    **Validates: Requirements 8.6, 8.8**
    """

    def test_serialization_round_trip_preserves_numeric_fields(self) -> None:
        """Metrics serialized to JSON and deserialized back preserve all numeric fields."""
        recorder = PerformanceRecorder(rank=1, stage=2)
        with recorder.span("prefill_forward", mode="prefill"):
            time.sleep(0.001)
        with recorder.span("decode_step", mode="decode"):
            time.sleep(0.001)
        recorder.increment_counter("tokens_generated", amount=42)

        json_output = recorder.serialize_to_json()
        events, counters, rank, stage = deserialize_events_from_json(json_output)

        assert rank == 1
        assert stage == 2
        assert len(events) == 2
        assert counters["tokens_generated"] == 42

        # Verify numeric fields are preserved exactly
        original_events = recorder.events
        for original, deserialized in zip(original_events, events, strict=True):
            assert original.start_time == deserialized.start_time
            assert original.end_time == deserialized.end_time
            assert original.duration == deserialized.duration

    def test_serialization_produces_valid_json(self) -> None:
        """Output is valid JSON."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("test"):
            pass
        json_output = recorder.serialize_to_json()
        parsed = json.loads(json_output)
        assert "events" in parsed
        assert "counters" in parsed

    def test_serialization_with_metadata(self) -> None:
        """Metadata dict survives serialization round trip."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("send", metadata={"bytes": 8192, "target_rank": 3}):
            pass

        json_output = recorder.serialize_to_json()
        events, _, _, _ = deserialize_events_from_json(json_output)

        assert events[0].metadata["bytes"] == 8192
        assert events[0].metadata["target_rank"] == 3

    def test_serialization_with_empty_recorder(self) -> None:
        """Empty recorder serializes and deserializes cleanly."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        json_output = recorder.serialize_to_json()
        events, counters, rank, stage = deserialize_events_from_json(json_output)
        assert events == []
        assert counters == {}
        assert rank == 0
        assert stage == 0

    def test_serialization_preserves_mode_classification(self) -> None:
        """Mode classification (prefill/decode) survives round trip."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("prefill_op", mode="prefill"):
            pass
        with recorder.span("decode_op", mode="decode"):
            pass

        json_output = recorder.serialize_to_json()
        events, _, _, _ = deserialize_events_from_json(json_output)

        assert events[0].mode == "prefill"
        assert events[1].mode == "decode"


# ===========================================================================
# Tests for PerformanceSummary
# ===========================================================================


class TestPerformanceSummary:
    """Test summary aggregation.

    **Validates: Requirements 8.1, 8.2**
    """

    def test_summary_with_events(self) -> None:
        """Summary correctly aggregates event statistics."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("operation_a", mode="prefill"):
            time.sleep(0.005)
        with recorder.span("operation_b", mode="decode"):
            time.sleep(0.005)
        with recorder.span("operation_b", mode="decode"):
            time.sleep(0.005)

        summary = recorder.summarize()
        assert summary.total_event_count == 3
        assert summary.total_duration > 0.0
        assert summary.mean_duration > 0.0
        assert summary.minimum_duration >= 0.0
        assert summary.maximum_duration >= summary.minimum_duration
        assert summary.events_by_name["operation_a"] == 1
        assert summary.events_by_name["operation_b"] == 2
        assert summary.events_by_mode["prefill"] == 1
        assert summary.events_by_mode["decode"] == 2

    def test_summary_is_frozen(self) -> None:
        """PerformanceSummary instances are immutable."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("test"):
            pass
        summary = recorder.summarize()
        with pytest.raises(ValidationError):
            summary.total_event_count = 999  # type: ignore[misc]

    def test_empty_summary(self) -> None:
        """Summary of empty recorder has zero values."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        summary = recorder.summarize()
        assert summary.total_event_count == 0
        assert summary.total_duration == 0.0
        assert summary.mean_duration == 0.0
        assert summary.minimum_duration == 0.0
        assert summary.maximum_duration == 0.0
        assert summary.events_by_name == {}
        assert summary.events_by_mode == {}


# ===========================================================================
# Tests for reset
# ===========================================================================


class TestPerformanceRecorderReset:
    """Test recorder reset clears all state."""

    def test_reset_clears_events(self) -> None:
        """Reset removes all recorded events."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("operation"):
            pass
        recorder.increment_counter("test_counter", amount=5)
        assert len(recorder.events) == 1
        assert recorder.get_counter("test_counter") == 5

        recorder.reset()
        assert len(recorder.events) == 0
        assert recorder.get_counter("test_counter") == 0


# ===========================================================================
# Tests for DECODE_FAST_PATH_COUNTERS registry
# ===========================================================================


DECODE_FAST_PATH_COUNTERS = _mod.DECODE_FAST_PATH_COUNTERS


class TestDecodeFastPathCounterRegistry:
    """Test the decode fast-path counter registry is complete and well-formed.

    **Validates: Requirements 8.1, 8.2**
    """

    def test_registry_contains_all_required_counters(self) -> None:
        """Registry documents all required decode fast-path counters."""
        required_counters = {
            "fast_path_activation_sends",
            "fast_path_activation_receives",
            "generic_activation_sends",
            "generic_activation_receives",
            "shape_metadata_messages",
            "fast_path_fallbacks",
            "protocol_renegotiations",
            "token_result_send_count",
            "token_result_receive_count",
        }
        assert required_counters.issubset(set(DECODE_FAST_PATH_COUNTERS.keys()))

    def test_registry_values_are_nonempty_strings(self) -> None:
        """Each counter has a non-empty description string."""
        for name, description in DECODE_FAST_PATH_COUNTERS.items():
            assert isinstance(description, str), f"{name} description is not a string"
            assert len(description) > 0, f"{name} has empty description"

    def test_registry_is_immutable_dict(self) -> None:
        """Registry is a Final dict (cannot be reassigned at module level)."""
        # We verify it's a dict with string keys and string values
        assert isinstance(DECODE_FAST_PATH_COUNTERS, dict)
        for key, value in DECODE_FAST_PATH_COUNTERS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_all_registered_counters_can_be_incremented(self) -> None:
        """All registered counter names work with PerformanceRecorder."""
        recorder = PerformanceRecorder(rank=0, stage=0)
        for counter_name in DECODE_FAST_PATH_COUNTERS:
            recorder.increment_counter(counter_name)
            assert recorder.get_counter(counter_name) == 1
