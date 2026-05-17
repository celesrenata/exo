"""
Unit tests for BatchedDecodeProtocol and BatchedActivationMessage.

Covers:
- BatchedDecodeProtocol is frozen and stores correct fields
- BatchedActivationMessage correctly represents variable batch membership
- Active slot count can vary between steps (protocol supports it)

Requirements: 2.11, 3.2, 3.3, 3.6, 3.10
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest  # noqa: I001

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
BatchedDecodeProtocol = _batching_mod.BatchedDecodeProtocol
BatchedActivationMessage = _batching_mod.BatchedActivationMessage


# ---------------------------------------------------------------------------
# Test: BatchedDecodeProtocol is frozen and stores correct fields
# ---------------------------------------------------------------------------


class TestBatchedDecodeProtocol:
    """Tests for BatchedDecodeProtocol immutability and field storage."""

    def test_creation_with_all_fields(self) -> None:
        """BatchedDecodeProtocol stores max_batch_size, hidden_size, active_slot_count."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=4,
            protocol_version=1,
        )
        assert protocol.max_batch_size == 8
        assert protocol.hidden_size == 3584
        assert protocol.active_slot_count == 4
        assert protocol.protocol_version == 1

    def test_default_protocol_version(self) -> None:
        """protocol_version defaults to 1 when not specified."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=4,
            hidden_size=2048,
            active_slot_count=2,
        )
        assert protocol.protocol_version == 1

    def test_frozen_cannot_modify_max_batch_size(self) -> None:
        """Attempting to modify max_batch_size raises FrozenInstanceError."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=4,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.max_batch_size = 16  # type: ignore[misc]

    def test_frozen_cannot_modify_hidden_size(self) -> None:
        """Attempting to modify hidden_size raises FrozenInstanceError."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=4,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.hidden_size = 4096  # type: ignore[misc]

    def test_frozen_cannot_modify_active_slot_count(self) -> None:
        """Attempting to modify active_slot_count raises FrozenInstanceError."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=4,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.active_slot_count = 6  # type: ignore[misc]

    def test_frozen_cannot_modify_protocol_version(self) -> None:
        """Attempting to modify protocol_version raises FrozenInstanceError."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=4,
            protocol_version=1,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.protocol_version = 2  # type: ignore[misc]

    def test_active_slot_count_zero(self) -> None:
        """Protocol supports zero active slots (empty batch)."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=0,
        )
        assert protocol.active_slot_count == 0
        assert protocol.max_batch_size == 8

    def test_active_slot_count_equals_max(self) -> None:
        """Protocol supports active_slot_count equal to max_batch_size (full batch)."""
        protocol = BatchedDecodeProtocol(
            max_batch_size=4,
            hidden_size=2048,
            active_slot_count=4,
        )
        assert protocol.active_slot_count == protocol.max_batch_size

    def test_different_hidden_sizes(self) -> None:
        """Protocol works with various hidden sizes matching real models."""
        for hidden_size in (2048, 3584, 4096, 5120):
            protocol = BatchedDecodeProtocol(
                max_batch_size=8,
                hidden_size=hidden_size,
                active_slot_count=3,
            )
            assert protocol.hidden_size == hidden_size


# ---------------------------------------------------------------------------
# Test: BatchedActivationMessage correctly represents variable batch membership
# ---------------------------------------------------------------------------


class TestBatchedActivationMessage:
    """Tests for BatchedActivationMessage variable batch membership."""

    def test_creation_with_all_fields(self) -> None:
        """BatchedActivationMessage stores active_slot_count, slot_indices, slot_generations."""
        message = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 1, 2),
            slot_generations=(1, 1, 1),
        )
        assert message.active_slot_count == 3
        assert message.slot_indices == (0, 1, 2)
        assert message.slot_generations == (1, 1, 1)

    def test_frozen_cannot_modify_active_slot_count(self) -> None:
        """Attempting to modify active_slot_count raises FrozenInstanceError."""
        message = BatchedActivationMessage(
            active_slot_count=2,
            slot_indices=(0, 1),
            slot_generations=(1, 1),
        )
        with pytest.raises(FrozenInstanceError):
            message.active_slot_count = 3  # type: ignore[misc]

    def test_frozen_cannot_modify_slot_indices(self) -> None:
        """Attempting to modify slot_indices raises FrozenInstanceError."""
        message = BatchedActivationMessage(
            active_slot_count=2,
            slot_indices=(0, 1),
            slot_generations=(1, 1),
        )
        with pytest.raises(FrozenInstanceError):
            message.slot_indices = (2, 3)  # type: ignore[misc]

    def test_frozen_cannot_modify_slot_generations(self) -> None:
        """Attempting to modify slot_generations raises FrozenInstanceError."""
        message = BatchedActivationMessage(
            active_slot_count=2,
            slot_indices=(0, 1),
            slot_generations=(1, 1),
        )
        with pytest.raises(FrozenInstanceError):
            message.slot_generations = (2, 2)  # type: ignore[misc]

    def test_slot_indices_length_matches_active_count(self) -> None:
        """slot_indices length equals active_slot_count."""
        message = BatchedActivationMessage(
            active_slot_count=4,
            slot_indices=(0, 2, 5, 7),
            slot_generations=(1, 3, 1, 2),
        )
        assert len(message.slot_indices) == message.active_slot_count

    def test_slot_generations_length_matches_active_count(self) -> None:
        """slot_generations length equals active_slot_count."""
        message = BatchedActivationMessage(
            active_slot_count=4,
            slot_indices=(0, 2, 5, 7),
            slot_generations=(1, 3, 1, 2),
        )
        assert len(message.slot_generations) == message.active_slot_count

    def test_non_contiguous_slot_indices(self) -> None:
        """slot_indices can be non-contiguous (gaps from completed requests)."""
        message = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 3, 7),
            slot_generations=(1, 2, 1),
        )
        assert message.slot_indices == (0, 3, 7)

    def test_empty_message(self) -> None:
        """BatchedActivationMessage supports zero active slots."""
        message = BatchedActivationMessage(
            active_slot_count=0,
            slot_indices=(),
            slot_generations=(),
        )
        assert message.active_slot_count == 0
        assert len(message.slot_indices) == 0
        assert len(message.slot_generations) == 0

    def test_single_active_slot(self) -> None:
        """BatchedActivationMessage works with a single active slot."""
        message = BatchedActivationMessage(
            active_slot_count=1,
            slot_indices=(5,),
            slot_generations=(3,),
        )
        assert message.active_slot_count == 1
        assert message.slot_indices[0] == 5
        assert message.slot_generations[0] == 3

    def test_different_generations_per_slot(self) -> None:
        """Different slots can have different generation counters."""
        message = BatchedActivationMessage(
            active_slot_count=4,
            slot_indices=(0, 1, 2, 3),
            slot_generations=(1, 5, 2, 8),
        )
        assert message.slot_generations == (1, 5, 2, 8)

    def test_staleness_detection_across_steps(self) -> None:
        """Slot generations enable staleness detection when slots are reused."""
        # Step 1: slots 0, 1, 2 active with generation 1
        msg_step_1 = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 1, 2),
            slot_generations=(1, 1, 1),
        )

        # Step 2: slot 1 was released and reassigned (generation incremented)
        msg_step_2 = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 1, 2),
            slot_generations=(1, 2, 1),
        )

        # Slot 1's generation changed — stale messages from gen 1 are detectable
        assert msg_step_1.slot_generations[1] != msg_step_2.slot_generations[1]
        assert msg_step_1.slot_generations[0] == msg_step_2.slot_generations[0]


# ---------------------------------------------------------------------------
# Test: Active slot count can vary between steps (protocol supports it)
# ---------------------------------------------------------------------------


class TestVariableActiveSlotCount:
    """Tests that the protocol supports varying active slot counts across steps."""

    def test_protocol_supports_growing_batch(self) -> None:
        """Active slot count can increase as new requests join."""
        # Step 1: 2 active requests
        protocol_step_1 = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=2,
        )

        # Step 2: 5 active requests (3 new requests admitted)
        protocol_step_2 = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=5,
        )

        # max_batch_size and hidden_size remain constant
        assert protocol_step_1.max_batch_size == protocol_step_2.max_batch_size
        assert protocol_step_1.hidden_size == protocol_step_2.hidden_size
        # active_slot_count varies
        assert protocol_step_1.active_slot_count < protocol_step_2.active_slot_count

    def test_protocol_supports_shrinking_batch(self) -> None:
        """Active slot count can decrease as requests complete."""
        # Step 1: 6 active requests
        protocol_step_1 = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=6,
        )

        # Step 2: 3 active requests (3 completed)
        protocol_step_2 = BatchedDecodeProtocol(
            max_batch_size=8,
            hidden_size=3584,
            active_slot_count=3,
        )

        assert protocol_step_1.max_batch_size == protocol_step_2.max_batch_size
        assert protocol_step_1.active_slot_count > protocol_step_2.active_slot_count

    def test_message_varies_with_batch_membership(self) -> None:
        """BatchedActivationMessage reflects changing batch membership."""
        # Step 1: requests in slots 0, 1, 2
        msg_1 = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 1, 2),
            slot_generations=(1, 1, 1),
        )

        # Step 2: request in slot 1 completed, new request in slot 4
        msg_2 = BatchedActivationMessage(
            active_slot_count=3,
            slot_indices=(0, 2, 4),
            slot_generations=(1, 1, 1),
        )

        # Same active count but different slot membership
        assert msg_1.active_slot_count == msg_2.active_slot_count
        assert msg_1.slot_indices != msg_2.slot_indices

    def test_protocol_and_message_consistent_max_batch(self) -> None:
        """Protocol max_batch_size bounds the message's slot_indices values."""
        max_batch = 8
        protocol = BatchedDecodeProtocol(
            max_batch_size=max_batch,
            hidden_size=3584,
            active_slot_count=4,
        )
        message = BatchedActivationMessage(
            active_slot_count=4,
            slot_indices=(0, 2, 5, 7),
            slot_generations=(1, 1, 2, 1),
        )

        # All slot indices must be within [0, max_batch_size)
        for idx in message.slot_indices:
            assert 0 <= idx < protocol.max_batch_size

    def test_full_lifecycle_grow_and_shrink(self) -> None:
        """Simulates a full lifecycle: empty → grow → full → shrink → empty."""
        max_batch = 4
        hidden = 2048

        # Empty
        p0 = BatchedDecodeProtocol(max_batch_size=max_batch, hidden_size=hidden, active_slot_count=0)
        assert p0.active_slot_count == 0

        # One request admitted
        p1 = BatchedDecodeProtocol(max_batch_size=max_batch, hidden_size=hidden, active_slot_count=1)
        assert p1.active_slot_count == 1

        # Full batch
        p2 = BatchedDecodeProtocol(max_batch_size=max_batch, hidden_size=hidden, active_slot_count=4)
        assert p2.active_slot_count == max_batch

        # Two requests complete
        p3 = BatchedDecodeProtocol(max_batch_size=max_batch, hidden_size=hidden, active_slot_count=2)
        assert p3.active_slot_count == 2

        # All requests complete
        p4 = BatchedDecodeProtocol(max_batch_size=max_batch, hidden_size=hidden, active_slot_count=0)
        assert p4.active_slot_count == 0

        # max_batch_size never changes
        for p in (p0, p1, p2, p3, p4):
            assert p.max_batch_size == max_batch
            assert p.hidden_size == hidden
