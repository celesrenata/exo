"""
Unit tests for RequestIdentifierMap.

Covers:
- Assign and release slots
- Generation counter increments on release
- Stale generation detection
- Free list reuse
- Full capacity handling (no free slots raises)
- Reverse lookup works correctly
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

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
RequestIdentifierMap = _batching_mod.RequestIdentifierMap


# ---------------------------------------------------------------------------
# Test: Assign and release slots
# ---------------------------------------------------------------------------


class TestAssignAndRelease:
    """Tests for basic slot assignment and release."""

    def test_assign_returns_slot_index_and_generation(self) -> None:
        """assign_slot returns a tuple of (slot_index, generation)."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, generation = id_map.assign_slot("req-1")

        assert isinstance(slot_index, int)
        assert isinstance(generation, int)
        assert 0 <= slot_index < 4
        assert generation == 1  # initial generation

    def test_assign_multiple_requests_get_different_slots(self) -> None:
        """Each request gets a unique slot index."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        slot_0, _ = id_map.assign_slot("req-a")
        slot_1, _ = id_map.assign_slot("req-b")
        slot_2, _ = id_map.assign_slot("req-c")

        assert len({slot_0, slot_1, slot_2}) == 3

    def test_release_makes_slot_available(self) -> None:
        """After release, the slot can be assigned to a new request."""
        id_map = RequestIdentifierMap(max_batch_size=1)

        id_map.assign_slot("req-1")
        assert id_map.available_slot_count == 0

        id_map.release_slot("req-1")
        assert id_map.available_slot_count == 1

        # Can assign again
        slot_index, _ = id_map.assign_slot("req-2")
        assert slot_index == 0

    def test_assign_duplicate_request_raises(self) -> None:
        """Assigning the same request ID twice raises ValueError."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        id_map.assign_slot("req-1")

        with pytest.raises(ValueError, match="already assigned"):
            id_map.assign_slot("req-1")

    def test_release_unknown_request_raises(self) -> None:
        """Releasing an unassigned request raises ValueError."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        with pytest.raises(ValueError, match="does not have an assigned slot"):
            id_map.release_slot("nonexistent")


# ---------------------------------------------------------------------------
# Test: Generation counter increments on release
# ---------------------------------------------------------------------------


class TestGenerationCounter:
    """Tests for slot generation counter behavior."""

    def test_initial_generation_is_one(self) -> None:
        """Slots start with generation 1."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        _, generation = id_map.assign_slot("req-1")
        assert generation == 1

    def test_generation_increments_on_release(self) -> None:
        """Each release increments the slot's generation counter."""
        id_map = RequestIdentifierMap(max_batch_size=1)

        _, gen1 = id_map.assign_slot("req-1")
        assert gen1 == 1

        id_map.release_slot("req-1")

        _, gen2 = id_map.assign_slot("req-2")
        assert gen2 == 2

        id_map.release_slot("req-2")

        _, gen3 = id_map.assign_slot("req-3")
        assert gen3 == 3

    def test_generation_is_per_slot(self) -> None:
        """Generation counters are independent per slot."""
        id_map = RequestIdentifierMap(max_batch_size=2)

        slot_a, gen_a = id_map.assign_slot("req-a")
        slot_b, gen_b = id_map.assign_slot("req-b")

        # Release and reassign slot_a multiple times
        id_map.release_slot("req-a")
        id_map.assign_slot("req-a2")
        id_map.release_slot("req-a2")
        _, gen_a3 = id_map.assign_slot("req-a3")

        # slot_a has been released twice, so generation is 3
        assert gen_a3 == 3

        # slot_b has never been released, still generation 1
        result = id_map.get_slot("req-b")
        assert result is not None
        assert result[1] == 1


# ---------------------------------------------------------------------------
# Test: Stale generation detection
# ---------------------------------------------------------------------------


class TestStaleGenerationDetection:
    """Tests for is_generation_current stale message detection."""

    def test_current_generation_returns_true(self) -> None:
        """is_generation_current returns True for matching generation."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, generation = id_map.assign_slot("req-1")

        assert id_map.is_generation_current(slot_index, generation) is True

    def test_stale_generation_returns_false(self) -> None:
        """is_generation_current returns False for old generation."""
        id_map = RequestIdentifierMap(max_batch_size=1)

        slot_index, old_generation = id_map.assign_slot("req-1")
        id_map.release_slot("req-1")

        # Slot now has generation 2, old message has generation 1
        assert id_map.is_generation_current(slot_index, old_generation) is False

    def test_future_generation_returns_false(self) -> None:
        """is_generation_current returns False for future generation."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, generation = id_map.assign_slot("req-1")

        assert id_map.is_generation_current(slot_index, generation + 1) is False

    def test_out_of_range_slot_raises(self) -> None:
        """is_generation_current raises IndexError for invalid slot."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        with pytest.raises(IndexError):
            id_map.is_generation_current(4, 1)

        with pytest.raises(IndexError):
            id_map.is_generation_current(-1, 1)


# ---------------------------------------------------------------------------
# Test: Free list reuse
# ---------------------------------------------------------------------------


class TestFreeListReuse:
    """Tests for slot reuse via the free list."""

    def test_released_slot_is_reused(self) -> None:
        """A released slot index is returned to the free list and reused."""
        id_map = RequestIdentifierMap(max_batch_size=2)

        slot_0, _ = id_map.assign_slot("req-1")
        slot_1, _ = id_map.assign_slot("req-2")

        # Release slot_0
        id_map.release_slot("req-1")

        # Next assignment reuses slot_0
        reused_slot, _ = id_map.assign_slot("req-3")
        assert reused_slot == slot_0

    def test_fifo_reuse_order(self) -> None:
        """Slots are reused in FIFO order (first released, first reused)."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        # Assign all 4 slots
        slots = []
        for i in range(4):
            slot, _ = id_map.assign_slot(f"req-{i}")
            slots.append(slot)

        # Release in order: slot 2, then slot 0
        id_map.release_slot("req-2")
        id_map.release_slot("req-0")

        # Reuse should follow FIFO: slot 2 first, then slot 0
        reused_1, _ = id_map.assign_slot("new-1")
        reused_2, _ = id_map.assign_slot("new-2")

        assert reused_1 == slots[2]
        assert reused_2 == slots[0]


# ---------------------------------------------------------------------------
# Test: Full capacity handling
# ---------------------------------------------------------------------------


class TestFullCapacity:
    """Tests for behavior when all slots are occupied."""

    def test_assign_when_full_raises(self) -> None:
        """Assigning when no free slots are available raises RuntimeError."""
        id_map = RequestIdentifierMap(max_batch_size=2)
        id_map.assign_slot("req-1")
        id_map.assign_slot("req-2")

        with pytest.raises(RuntimeError, match="No free slots available"):
            id_map.assign_slot("req-3")

    def test_assign_after_release_from_full(self) -> None:
        """After releasing from a full map, assignment succeeds."""
        id_map = RequestIdentifierMap(max_batch_size=2)
        id_map.assign_slot("req-1")
        id_map.assign_slot("req-2")

        id_map.release_slot("req-1")
        # Now one slot is free
        slot, gen = id_map.assign_slot("req-3")
        assert gen == 2  # generation incremented from release

    def test_active_and_available_counts(self) -> None:
        """active_slot_count and available_slot_count sum to max_batch_size."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        assert id_map.active_slot_count == 0
        assert id_map.available_slot_count == 4

        id_map.assign_slot("req-1")
        assert id_map.active_slot_count == 1
        assert id_map.available_slot_count == 3

        id_map.assign_slot("req-2")
        id_map.assign_slot("req-3")
        assert id_map.active_slot_count == 3
        assert id_map.available_slot_count == 1

        id_map.release_slot("req-2")
        assert id_map.active_slot_count == 2
        assert id_map.available_slot_count == 2


# ---------------------------------------------------------------------------
# Test: Reverse lookup
# ---------------------------------------------------------------------------


class TestReverseLookup:
    """Tests for get_request_id reverse lookup."""

    def test_reverse_lookup_occupied_slot(self) -> None:
        """get_request_id returns the request ID for an occupied slot."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, _ = id_map.assign_slot("req-abc")

        assert id_map.get_request_id(slot_index) == "req-abc"

    def test_reverse_lookup_empty_slot(self) -> None:
        """get_request_id returns None for an empty slot."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        assert id_map.get_request_id(0) is None

    def test_reverse_lookup_after_release(self) -> None:
        """get_request_id returns None after the slot is released."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, _ = id_map.assign_slot("req-1")

        id_map.release_slot("req-1")
        assert id_map.get_request_id(slot_index) is None

    def test_reverse_lookup_after_reassignment(self) -> None:
        """get_request_id returns the new request after reassignment."""
        id_map = RequestIdentifierMap(max_batch_size=1)

        id_map.assign_slot("req-old")
        id_map.release_slot("req-old")
        id_map.assign_slot("req-new")

        assert id_map.get_request_id(0) == "req-new"

    def test_reverse_lookup_out_of_range_raises(self) -> None:
        """get_request_id raises IndexError for invalid slot index."""
        id_map = RequestIdentifierMap(max_batch_size=4)

        with pytest.raises(IndexError):
            id_map.get_request_id(4)

        with pytest.raises(IndexError):
            id_map.get_request_id(-1)


# ---------------------------------------------------------------------------
# Test: Forward lookup (get_slot)
# ---------------------------------------------------------------------------


class TestForwardLookup:
    """Tests for get_slot forward lookup."""

    def test_get_slot_assigned_request(self) -> None:
        """get_slot returns (slot_index, generation) for assigned request."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        slot_index, generation = id_map.assign_slot("req-1")

        result = id_map.get_slot("req-1")
        assert result == (slot_index, generation)

    def test_get_slot_unknown_request(self) -> None:
        """get_slot returns None for unknown request."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        assert id_map.get_slot("nonexistent") is None

    def test_get_slot_after_release(self) -> None:
        """get_slot returns None after the request is released."""
        id_map = RequestIdentifierMap(max_batch_size=4)
        id_map.assign_slot("req-1")
        id_map.release_slot("req-1")

        assert id_map.get_slot("req-1") is None


# ---------------------------------------------------------------------------
# Test: Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Tests for constructor parameter validation."""

    def test_zero_max_batch_size_raises(self) -> None:
        """max_batch_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RequestIdentifierMap(max_batch_size=0)

    def test_negative_max_batch_size_raises(self) -> None:
        """Negative max_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RequestIdentifierMap(max_batch_size=-1)

    def test_single_slot_works(self) -> None:
        """max_batch_size of 1 works correctly."""
        id_map = RequestIdentifierMap(max_batch_size=1)

        slot, gen = id_map.assign_slot("only-one")
        assert slot == 0
        assert gen == 1
        assert id_map.active_slot_count == 1
        assert id_map.available_slot_count == 0

        id_map.release_slot("only-one")
        assert id_map.active_slot_count == 0
        assert id_map.available_slot_count == 1
