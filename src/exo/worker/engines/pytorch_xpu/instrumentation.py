"""
Performance instrumentation for pipeline-parallel XPU inference.

Provides a context-manager span API using time.perf_counter() for high-resolution
timing, JSON serialization for cross-run comparison, and per-rank per-stage per-mode
metrics with low overhead in normal mode.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 8.10**
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Final, Generator, Literal, final

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

EventMode = Literal["prefill", "decode"]
"""Classification of an event as either prefill or decode phase."""


# ---------------------------------------------------------------------------
# Decode Fast-Path Counter Registry
# ---------------------------------------------------------------------------

DECODE_FAST_PATH_COUNTERS: Final[dict[str, str]] = {
    "fast_path_activation_sends": (
        "Number of decode activation tensors sent via the fast path "
        "(no per-token shape metadata). Incremented in "
        "_send_decode_activation_rank0() and _send_decode_activation_worker()."
    ),
    "fast_path_activation_receives": (
        "Number of decode activation tensors received via the fast path "
        "(no per-token shape metadata). Incremented in "
        "_recv_decode_activation_worker()."
    ),
    "generic_activation_sends": (
        "Number of activation tensors sent via the generic path "
        "(includes shape metadata). Incremented when the fast path is "
        "unavailable or falls back."
    ),
    "generic_activation_receives": (
        "Number of activation tensors received via the generic path "
        "(includes shape metadata). Incremented when the fast path is "
        "unavailable or falls back."
    ),
    "shape_metadata_messages": (
        "Number of shape metadata tensors sent or received. Each call to "
        "_send_with_shape() or _recv_with_shape() that transmits the seq_len "
        "tensor increments this counter."
    ),
    "fast_path_fallbacks": (
        "Number of times the fast path failed and fell back to the generic "
        "path. Incremented on fast-path send/receive exceptions or protocol "
        "mismatch."
    ),
    "protocol_renegotiations": (
        "Number of protocol renegotiation events. Currently always 0 because "
        "renegotiation is not yet implemented — the protocol is negotiated "
        "once after prefill and remains fixed. This counter exists for future "
        "use when dynamic microbatch size changes trigger renegotiation."
    ),
    "token_result_send_count": (
        "Number of token result packets sent from the final rank to rank 0 "
        "via point-to-point communication. Incremented in "
        "send_token_results_to_rank_zero()."
    ),
    "token_result_receive_count": (
        "Number of token result packets received on rank 0 from the final "
        "rank via point-to-point communication. Incremented in "
        "receive_token_results_from_final_rank()."
    ),
}
"""Registry of all decode fast-path instrumentation counter names and their meanings.

After a generation run, these counters provide complete observability of the
communication path:
- How many messages used the fast path vs generic path
- How many shape metadata messages were sent
- How many fallbacks occurred
- How many token results were exchanged via point-to-point

Use ``PerformanceRecorder.get_counter(name)`` to query individual counters,
or ``PerformanceRecorder.summarize().counters`` for the full snapshot.
"""


# ---------------------------------------------------------------------------
# PerformanceEvent — immutable record of a single timed span
# ---------------------------------------------------------------------------


@final
class PerformanceEvent(BaseModel):
    """
    Immutable record of a single timed span in the pipeline.

    Each event captures a named operation with start/end timestamps,
    duration, optional parent reference for nesting, mode classification,
    rank, stage, and arbitrary metadata.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    event_name: str
    """Descriptive name of the operation being measured."""

    start_time: float
    """Absolute start time from time.perf_counter()."""

    end_time: float
    """Absolute end time from time.perf_counter()."""

    duration: float
    """Elapsed time in seconds (end_time - start_time)."""

    parent_event_name: str | None = None
    """Name of the parent event if this span is nested."""

    mode: EventMode | None = None
    """Whether this event occurred during prefill or decode phase."""

    rank: int | None = None
    """Distributed rank that recorded this event."""

    stage: int | None = None
    """Pipeline stage index that recorded this event."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary key-value metadata attached to this event."""


# ---------------------------------------------------------------------------
# PerformanceSummary — aggregated metrics from a recording session
# ---------------------------------------------------------------------------


@final
class PerformanceSummary(BaseModel):
    """
    Aggregated metrics from a performance recording session.

    Contains event counts, timing statistics, and counter values
    suitable for cross-run comparison when serialized to JSON.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    total_event_count: int
    """Total number of events recorded."""

    total_duration: float
    """Sum of all event durations in seconds."""

    mean_duration: float
    """Mean event duration in seconds."""

    minimum_duration: float
    """Minimum event duration in seconds."""

    maximum_duration: float
    """Maximum event duration in seconds."""

    events_by_name: dict[str, int]
    """Count of events grouped by event_name."""

    events_by_mode: dict[str, int]
    """Count of events grouped by mode (prefill/decode)."""

    counters: dict[str, int]
    """Accumulated counter values."""

    rank: int | None = None
    """Rank this summary pertains to, if applicable."""

    stage: int | None = None
    """Stage this summary pertains to, if applicable."""


# ---------------------------------------------------------------------------
# PerformanceRecorder — mutable collector with context-manager span API
# ---------------------------------------------------------------------------

_EMPTY_METADATA: Final[dict[str, Any]] = {}


@final
class PerformanceRecorder:
    """
    Mutable collector of performance events with a context-manager span API.

    Provides:
    - Context-manager spans using time.perf_counter() for high-resolution timing
    - Event collection with parent-child nesting support
    - Named counters for counting discrete occurrences
    - JSON export for cross-run comparison
    - Low overhead: when disabled, span() is a near-no-op

    Example:
        recorder = PerformanceRecorder(rank=0, stage=0)
        with recorder.span("prefill_forward", mode="prefill") as event_builder:
            with recorder.span("layer_0_compute", mode="prefill", parent="prefill_forward"):
                do_layer_compute()
        summary = recorder.summarize()
        json_output = recorder.serialize_to_json()
    """

    __slots__ = (
        "_enabled",
        "_rank",
        "_stage",
        "_events",
        "_counters",
        "_active_span_stack",
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        rank: int | None = None,
        stage: int | None = None,
    ) -> None:
        self._enabled: bool = enabled
        self._rank: int | None = rank
        self._stage: int | None = stage
        self._events: list[PerformanceEvent] = []
        self._counters: dict[str, int] = {}
        self._active_span_stack: list[str] = []

    @property
    def enabled(self) -> bool:
        """Whether recording is active."""
        return self._enabled

    @property
    def rank(self) -> int | None:
        """Distributed rank for this recorder."""
        return self._rank

    @property
    def stage(self) -> int | None:
        """Pipeline stage for this recorder."""
        return self._stage

    @property
    def events(self) -> list[PerformanceEvent]:
        """All recorded events (read-only view)."""
        return list(self._events)

    @property
    def counters(self) -> dict[str, int]:
        """All counter values (read-only copy)."""
        return dict(self._counters)

    @contextmanager
    def span(
        self,
        event_name: str,
        *,
        mode: EventMode | None = None,
        parent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that records a timed span.

        When the recorder is disabled, this is a near-no-op that yields
        immediately without measuring time.

        Args:
            event_name: Descriptive name for this span.
            mode: Whether this is a prefill or decode operation.
            parent: Explicit parent event name for nesting. If None and
                    there is an active span on the stack, the top of the
                    stack is used as the implicit parent.
            metadata: Optional key-value pairs to attach to the event.

        Yields:
            None — the span is automatically closed when the context exits.
        """
        if not self._enabled:
            yield
            return

        # Determine parent: explicit > implicit from stack > None
        resolved_parent = parent
        if resolved_parent is None and self._active_span_stack:
            resolved_parent = self._active_span_stack[-1]

        self._active_span_stack.append(event_name)
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._active_span_stack.pop()
            duration = end - start
            event = PerformanceEvent(
                event_name=event_name,
                start_time=start,
                end_time=end,
                duration=duration,
                parent_event_name=resolved_parent,
                mode=mode,
                rank=self._rank,
                stage=self._stage,
                metadata=metadata if metadata is not None else _EMPTY_METADATA,
            )
            self._events.append(event)

    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """
        Increment a named counter by the given amount.

        Counters track discrete occurrences (allocations, messages sent,
        protocol renegotiations, etc.).

        Args:
            counter_name: Name of the counter to increment.
            amount: Amount to add (must be positive).
        """
        if not self._enabled:
            return
        current = self._counters.get(counter_name, 0)
        self._counters[counter_name] = current + amount

    def get_counter(self, counter_name: str) -> int:
        """
        Get the current value of a named counter.

        Returns 0 if the counter has never been incremented.
        """
        return self._counters.get(counter_name, 0)

    def summarize(self) -> PerformanceSummary:
        """
        Produce an aggregated summary of all recorded events and counters.

        Returns:
            PerformanceSummary with timing statistics and counter values.
        """
        total_count = len(self._events)

        if total_count == 0:
            return PerformanceSummary(
                total_event_count=0,
                total_duration=0.0,
                mean_duration=0.0,
                minimum_duration=0.0,
                maximum_duration=0.0,
                events_by_name={},
                events_by_mode={},
                counters=dict(self._counters),
                rank=self._rank,
                stage=self._stage,
            )

        durations = [event.duration for event in self._events]
        total_duration = sum(durations)
        mean_duration = total_duration / total_count
        minimum_duration = min(durations)
        maximum_duration = max(durations)

        events_by_name: dict[str, int] = {}
        events_by_mode: dict[str, int] = {}

        for event in self._events:
            events_by_name[event.event_name] = (
                events_by_name.get(event.event_name, 0) + 1
            )
            if event.mode is not None:
                events_by_mode[event.mode] = events_by_mode.get(event.mode, 0) + 1

        return PerformanceSummary(
            total_event_count=total_count,
            total_duration=total_duration,
            mean_duration=mean_duration,
            minimum_duration=minimum_duration,
            maximum_duration=maximum_duration,
            events_by_name=events_by_name,
            events_by_mode=events_by_mode,
            counters=dict(self._counters),
            rank=self._rank,
            stage=self._stage,
        )

    def serialize_to_json(self) -> str:
        """
        Serialize all events and counters to a JSON string.

        The output preserves all numeric fields for cross-run comparison.
        Deserialization via deserialize_from_json() round-trips all values.

        Returns:
            JSON string containing events and counters.
        """
        payload: dict[str, Any] = {
            "rank": self._rank,
            "stage": self._stage,
            "events": [event.model_dump() for event in self._events],
            "counters": self._counters,
        }
        return json.dumps(payload, allow_nan=False)

    def reset(self) -> None:
        """Clear all recorded events and counters."""
        self._events.clear()
        self._counters.clear()
        self._active_span_stack.clear()


# ---------------------------------------------------------------------------
# Deserialization — reconstruct from JSON for cross-run comparison
# ---------------------------------------------------------------------------


def deserialize_events_from_json(
    json_string: str,
) -> tuple[list[PerformanceEvent], dict[str, int], int | None, int | None]:
    """
    Deserialize a JSON string produced by PerformanceRecorder.serialize_to_json().

    Returns:
        Tuple of (events, counters, rank, stage).
    """
    payload: dict[str, Any] = json.loads(json_string)  # pyright: ignore[reportAny]
    raw_events = list[dict[str, Any]](payload["events"])  # pyright: ignore[reportAny]
    events: list[PerformanceEvent] = [
        PerformanceEvent.model_validate(event_data)
        for event_data in raw_events
    ]
    counters = dict[str, int](payload["counters"])  # pyright: ignore[reportAny]
    rank: int | None = payload.get("rank")
    stage: int | None = payload.get("stage")
    return events, counters, rank, stage
