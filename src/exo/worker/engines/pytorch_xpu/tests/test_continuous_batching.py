"""
Unit tests for ContinuousBatchScheduler.

Covers:
- Admit → prefill → decode → complete lifecycle
- Cancel from any state
- Multiple concurrent requests in decode-ready queue
- Properties reflect correct counts
- get_decode_batch returns all decode-ready requests
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Load modules with torch mocked out
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_SAMPLING_PATH = _ENGINE_DIR / "sampling.py"
_BATCHING_PATH = _ENGINE_DIR / "continuous_batching.py"


def _ensure_torch_mock() -> None:
    """Ensure a minimal torch mock is in sys.modules."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "Tensor"):
        torch_mock = types.ModuleType("torch")
        torch_mock.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_mock


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load a module by file path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ensure_torch_mock()

# Load sampling first (dependency of continuous_batching)
_sampling_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.sampling", _SAMPLING_PATH
)
SamplingConfiguration = _sampling_mod.SamplingConfiguration

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
ContinuousBatchScheduler = _batching_mod.ContinuousBatchScheduler
SchedulerMetrics = _batching_mod.SchedulerMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = SamplingConfiguration(do_sample=False)


def _admit(
    scheduler: ContinuousBatchScheduler,
    request_id: str,
    prompt_tokens: list[int] | None = None,
) -> None:
    """Shorthand to admit a request with default config."""
    scheduler.admit_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens or [1, 2, 3],
        sampling_config=_DEFAULT_CONFIG,
    )


# ---------------------------------------------------------------------------
# Test: Full lifecycle (admit → prefill → decode → complete)
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """Tests for the complete request lifecycle."""

    def test_admit_to_complete(self) -> None:
        """A request moves through all stages to completion."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")

        assert scheduler.get_request_state("req-1") == "admitted"
        assert scheduler.active_request_count == 1
        assert scheduler.admission_queue_size == 1

        # Start prefill
        next_prefill = scheduler.get_next_prefill()
        assert next_prefill == "req-1"
        assert scheduler.get_request_state("req-1") == "prefilling"
        assert scheduler.admission_queue_size == 0

        # Complete prefill → decode ready
        scheduler.mark_prefill_complete("req-1")
        assert scheduler.get_request_state("req-1") == "decoding"
        assert scheduler.decode_ready_count == 1

        # Decode batch includes the request
        batch = scheduler.get_decode_batch()
        assert "req-1" in batch

        # Complete the request
        scheduler.mark_completed("req-1")
        assert scheduler.get_request_state("req-1") == "completed"
        assert scheduler.active_request_count == 0
        assert scheduler.completed_count == 1
        assert not scheduler.is_active("req-1")

    def test_admit_preserves_fifo_order(self) -> None:
        """Requests are prefilled in FIFO admission order."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-a")
        _admit(scheduler, "req-b")
        _admit(scheduler, "req-c")

        assert scheduler.get_next_prefill() == "req-a"
        scheduler.mark_prefill_complete("req-a")

        assert scheduler.get_next_prefill() == "req-b"
        scheduler.mark_prefill_complete("req-b")

        assert scheduler.get_next_prefill() == "req-c"
        scheduler.mark_prefill_complete("req-c")

    def test_only_one_prefill_at_a_time(self) -> None:
        """get_next_prefill returns None while a prefill is in progress."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        _admit(scheduler, "req-2")

        scheduler.get_next_prefill()  # starts req-1 prefill
        assert scheduler.get_next_prefill() is None  # blocked

        scheduler.mark_prefill_complete("req-1")
        assert scheduler.get_next_prefill() == "req-2"  # now available


# ---------------------------------------------------------------------------
# Test: Cancellation from any state
# ---------------------------------------------------------------------------


class TestCancellation:
    """Tests for request cancellation from various states."""

    def test_cancel_from_admitted(self) -> None:
        """Cancel a request that is waiting in the admission queue."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")

        scheduler.cancel_request("req-1")
        assert scheduler.get_request_state("req-1") == "cancelled"
        assert scheduler.active_request_count == 0
        assert scheduler.cancelled_count == 1
        assert scheduler.admission_queue_size == 0

    def test_cancel_from_prefilling(self) -> None:
        """Cancel a request that is currently being prefilled."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()

        scheduler.cancel_request("req-1")
        assert scheduler.get_request_state("req-1") == "cancelled"
        assert scheduler.active_request_count == 0
        assert scheduler.cancelled_count == 1
        # Prefill slot is freed
        _admit(scheduler, "req-2")
        assert scheduler.get_next_prefill() == "req-2"

    def test_cancel_from_decoding(self) -> None:
        """Cancel a request that is in the decode-ready queue."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()
        scheduler.mark_prefill_complete("req-1")

        scheduler.cancel_request("req-1")
        assert scheduler.get_request_state("req-1") == "cancelled"
        assert scheduler.decode_ready_count == 0
        assert scheduler.active_request_count == 0

    def test_cancel_nonexistent_raises(self) -> None:
        """Cancelling an unknown request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        with pytest.raises(ValueError, match="not active"):
            scheduler.cancel_request("nonexistent")

    def test_cancel_already_completed_raises(self) -> None:
        """Cancelling an already-completed request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()
        scheduler.mark_prefill_complete("req-1")
        scheduler.mark_completed("req-1")

        with pytest.raises(ValueError, match="not active"):
            scheduler.cancel_request("req-1")


# ---------------------------------------------------------------------------
# Test: Multiple concurrent requests in decode-ready queue
# ---------------------------------------------------------------------------


class TestMultipleConcurrentRequests:
    """Tests for multiple requests in the decode-ready queue."""

    def test_multiple_decode_ready(self) -> None:
        """Multiple requests can be decode-ready simultaneously."""
        scheduler = ContinuousBatchScheduler()

        # Admit and prefill three requests
        for i in range(3):
            _admit(scheduler, f"req-{i}")

        for i in range(3):
            prefill_id = scheduler.get_next_prefill()
            assert prefill_id == f"req-{i}"
            scheduler.mark_prefill_complete(f"req-{i}")

        assert scheduler.decode_ready_count == 3
        batch = scheduler.get_decode_batch()
        assert set(batch) == {"req-0", "req-1", "req-2"}

    def test_completing_one_does_not_affect_others(self) -> None:
        """Completing one request leaves others in decode-ready."""
        scheduler = ContinuousBatchScheduler()

        for i in range(3):
            _admit(scheduler, f"req-{i}")
            prefill_id = scheduler.get_next_prefill()
            assert prefill_id is not None
            scheduler.mark_prefill_complete(prefill_id)

        scheduler.mark_completed("req-1")
        batch = scheduler.get_decode_batch()
        assert set(batch) == {"req-0", "req-2"}
        assert scheduler.active_request_count == 2

    def test_admission_while_decoding(self) -> None:
        """New requests can be admitted while others are decoding."""
        scheduler = ContinuousBatchScheduler()

        # First request goes through to decode
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()
        scheduler.mark_prefill_complete("req-1")

        # Admit another while req-1 is decoding
        _admit(scheduler, "req-2")
        assert scheduler.admission_queue_size == 1
        assert scheduler.decode_ready_count == 1

        # Prefill req-2
        prefill_id = scheduler.get_next_prefill()
        assert prefill_id == "req-2"
        scheduler.mark_prefill_complete("req-2")

        # Both in decode batch
        batch = scheduler.get_decode_batch()
        assert set(batch) == {"req-1", "req-2"}


# ---------------------------------------------------------------------------
# Test: Properties reflect correct counts
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests that properties and metrics reflect correct state."""

    def test_empty_scheduler(self) -> None:
        """Empty scheduler has all zero counts."""
        scheduler = ContinuousBatchScheduler()
        assert scheduler.active_request_count == 0
        assert scheduler.decode_ready_count == 0
        assert scheduler.admission_queue_size == 0
        assert scheduler.completed_count == 0
        assert scheduler.cancelled_count == 0

    def test_counts_during_lifecycle(self) -> None:
        """Counts update correctly through the lifecycle."""
        scheduler = ContinuousBatchScheduler()

        _admit(scheduler, "req-1")
        assert scheduler.active_request_count == 1
        assert scheduler.admission_queue_size == 1

        scheduler.get_next_prefill()
        assert scheduler.active_request_count == 1
        assert scheduler.admission_queue_size == 0

        scheduler.mark_prefill_complete("req-1")
        assert scheduler.active_request_count == 1
        assert scheduler.decode_ready_count == 1

        scheduler.mark_completed("req-1")
        assert scheduler.active_request_count == 0
        assert scheduler.decode_ready_count == 0
        assert scheduler.completed_count == 1

    def test_metrics_snapshot(self) -> None:
        """get_metrics returns a frozen snapshot of current state."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        _admit(scheduler, "req-2")
        scheduler.get_next_prefill()
        scheduler.mark_prefill_complete("req-1")

        metrics = scheduler.get_metrics()
        assert isinstance(metrics, SchedulerMetrics)
        assert metrics.active_requests == 2
        assert metrics.admission_queue_size == 1
        assert metrics.prefilling_count == 0
        assert metrics.decode_ready_count == 1
        assert metrics.completed_count == 0
        assert metrics.cancelled_count == 0

    def test_metrics_with_prefilling(self) -> None:
        """Metrics correctly report prefilling_count."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()

        metrics = scheduler.get_metrics()
        assert metrics.prefilling_count == 1
        assert metrics.admission_queue_size == 0


# ---------------------------------------------------------------------------
# Test: get_decode_batch returns all decode-ready requests
# ---------------------------------------------------------------------------


class TestDecodeBatch:
    """Tests for decode batch formation."""

    def test_empty_batch_when_no_decode_ready(self) -> None:
        """get_decode_batch returns empty list when no requests are ready."""
        scheduler = ContinuousBatchScheduler()
        assert scheduler.get_decode_batch() == []

    def test_batch_contains_all_decode_ready(self) -> None:
        """get_decode_batch returns all decode-ready request IDs."""
        scheduler = ContinuousBatchScheduler()

        for i in range(5):
            _admit(scheduler, f"req-{i}")
            prefill_id = scheduler.get_next_prefill()
            assert prefill_id is not None
            scheduler.mark_prefill_complete(prefill_id)

        batch = scheduler.get_decode_batch()
        assert len(batch) == 5
        assert set(batch) == {f"req-{i}" for i in range(5)}

    def test_batch_excludes_completed(self) -> None:
        """Completed requests are not in the decode batch."""
        scheduler = ContinuousBatchScheduler()

        for i in range(3):
            _admit(scheduler, f"req-{i}")
            prefill_id = scheduler.get_next_prefill()
            assert prefill_id is not None
            scheduler.mark_prefill_complete(prefill_id)

        scheduler.mark_completed("req-0")
        scheduler.mark_completed("req-2")

        batch = scheduler.get_decode_batch()
        assert batch == ["req-1"]

    def test_batch_excludes_cancelled(self) -> None:
        """Cancelled requests are not in the decode batch."""
        scheduler = ContinuousBatchScheduler()

        for i in range(3):
            _admit(scheduler, f"req-{i}")
            prefill_id = scheduler.get_next_prefill()
            assert prefill_id is not None
            scheduler.mark_prefill_complete(prefill_id)

        scheduler.cancel_request("req-1")

        batch = scheduler.get_decode_batch()
        assert set(batch) == {"req-0", "req-2"}


# ---------------------------------------------------------------------------
# Test: Error conditions
# ---------------------------------------------------------------------------


class TestErrorConditions:
    """Tests for error handling and edge cases."""

    def test_duplicate_admission_raises(self) -> None:
        """Admitting the same request ID twice raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")

        with pytest.raises(ValueError, match="already active"):
            _admit(scheduler, "req-1")

    def test_readmit_completed_raises(self) -> None:
        """Re-admitting a completed request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.get_next_prefill()
        scheduler.mark_prefill_complete("req-1")
        scheduler.mark_completed("req-1")

        with pytest.raises(ValueError, match="already completed"):
            _admit(scheduler, "req-1")

    def test_readmit_cancelled_raises(self) -> None:
        """Re-admitting a cancelled request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        scheduler.cancel_request("req-1")

        with pytest.raises(ValueError, match="already been cancelled"):
            _admit(scheduler, "req-1")

    def test_mark_prefill_complete_wrong_request_raises(self) -> None:
        """Marking prefill complete for wrong request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        _admit(scheduler, "req-1")
        _admit(scheduler, "req-2")
        scheduler.get_next_prefill()  # starts req-1

        with pytest.raises(ValueError, match="not currently prefilling"):
            scheduler.mark_prefill_complete("req-2")

    def test_mark_completed_nonexistent_raises(self) -> None:
        """Completing an unknown request raises ValueError."""
        scheduler = ContinuousBatchScheduler()
        with pytest.raises(ValueError, match="not active"):
            scheduler.mark_completed("nonexistent")

    def test_get_next_prefill_empty_queue(self) -> None:
        """get_next_prefill returns None when admission queue is empty."""
        scheduler = ContinuousBatchScheduler()
        assert scheduler.get_next_prefill() is None

    def test_is_active_unknown_request(self) -> None:
        """is_active returns False for unknown request IDs."""
        scheduler = ContinuousBatchScheduler()
        assert not scheduler.is_active("unknown")

    def test_get_request_state_unknown(self) -> None:
        """get_request_state returns None for unknown request IDs."""
        scheduler = ContinuousBatchScheduler()
        assert scheduler.get_request_state("unknown") is None

    def test_get_sampling_config_active(self) -> None:
        """get_sampling_config returns config for active requests."""
        scheduler = ContinuousBatchScheduler()
        config = SamplingConfiguration(do_sample=True, top_k=50)
        scheduler.admit_request("req-1", [1, 2, 3], config)

        result = scheduler.get_sampling_config("req-1")
        assert result is not None
        assert result.top_k == 50

    def test_get_sampling_config_inactive(self) -> None:
        """get_sampling_config returns None for inactive requests."""
        scheduler = ContinuousBatchScheduler()
        assert scheduler.get_sampling_config("unknown") is None
