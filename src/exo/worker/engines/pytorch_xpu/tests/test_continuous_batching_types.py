"""
Unit tests for continuous batching state types.

Covers:
- RequestRuntimeState is mutable and tracks token generation
- BatchSlotState is frozen (immutable)
- DecodeMicrobatch correctly represents batch composition
- TokenResultBatch is frozen and contains matching-length tuples
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
SamplingConfiguration = _sampling_mod.SamplingConfiguration

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
RequestRuntimeState = _batching_mod.RequestRuntimeState
BatchSlotState = _batching_mod.BatchSlotState
DecodeMicrobatch = _batching_mod.DecodeMicrobatch
TokenResultBatch = _batching_mod.TokenResultBatch


# ---------------------------------------------------------------------------
# Test: RequestRuntimeState is mutable and tracks token generation
# ---------------------------------------------------------------------------


class TestRequestRuntimeState:
    """Tests for RequestRuntimeState mutability and token tracking."""

    def test_creation_with_all_fields(self) -> None:
        """RequestRuntimeState can be created with all required fields."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=0,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=100,
            is_finished=False,
            max_tokens=50,
            sampling_config=config,
        )
        assert state.request_id == "req-1"
        assert state.slot_index == 0
        assert state.slot_generation == 1
        assert state.tokens_generated == 0
        assert state.last_token_id == 100
        assert state.is_finished is False
        assert state.max_tokens == 50
        assert state.sampling_config is config

    def test_mutable_tokens_generated(self) -> None:
        """tokens_generated can be updated in place."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=0,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=100,
            is_finished=False,
            max_tokens=50,
            sampling_config=config,
        )
        state.tokens_generated = 5
        assert state.tokens_generated == 5

    def test_mutable_last_token_id(self) -> None:
        """last_token_id can be updated in place."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=0,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=100,
            is_finished=False,
            max_tokens=50,
            sampling_config=config,
        )
        state.last_token_id = 200
        assert state.last_token_id == 200

    def test_mutable_is_finished(self) -> None:
        """is_finished can be updated in place."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=0,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=100,
            is_finished=False,
            max_tokens=50,
            sampling_config=config,
        )
        state.is_finished = True
        assert state.is_finished is True

    def test_tracks_token_generation_progress(self) -> None:
        """Simulates token generation progress through field updates."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=2,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=1,  # BOS token
            is_finished=False,
            max_tokens=3,
            sampling_config=config,
        )

        # Generate first token
        state.last_token_id = 42
        state.tokens_generated = 1
        assert state.tokens_generated == 1
        assert state.last_token_id == 42
        assert state.is_finished is False

        # Generate second token
        state.last_token_id = 99
        state.tokens_generated = 2
        assert state.tokens_generated == 2
        assert state.last_token_id == 99

        # Generate third token (max_tokens reached)
        state.last_token_id = 7
        state.tokens_generated = 3
        state.is_finished = True
        assert state.tokens_generated == 3
        assert state.is_finished is True

    def test_slot_generation_mutable(self) -> None:
        """slot_generation can be incremented for slot reuse."""
        config = SamplingConfiguration(do_sample=False)
        state = RequestRuntimeState(
            request_id="req-1",
            slot_index=0,
            slot_generation=1,
            tokens_generated=0,
            last_token_id=0,
            is_finished=False,
            max_tokens=10,
            sampling_config=config,
        )
        state.slot_generation = 2
        assert state.slot_generation == 2


# ---------------------------------------------------------------------------
# Test: BatchSlotState is frozen (immutable)
# ---------------------------------------------------------------------------


class TestBatchSlotState:
    """Tests for BatchSlotState immutability."""

    def test_creation_active_slot(self) -> None:
        """BatchSlotState can represent an active slot with a request."""
        slot = BatchSlotState(
            slot_index=0,
            request_id="req-1",
            slot_generation=1,
            is_active=True,
        )
        assert slot.slot_index == 0
        assert slot.request_id == "req-1"
        assert slot.slot_generation == 1
        assert slot.is_active is True

    def test_creation_empty_slot(self) -> None:
        """BatchSlotState can represent an empty slot."""
        slot = BatchSlotState(
            slot_index=3,
            request_id=None,
            slot_generation=0,
            is_active=False,
        )
        assert slot.slot_index == 3
        assert slot.request_id is None
        assert slot.slot_generation == 0
        assert slot.is_active is False

    def test_frozen_cannot_modify_slot_index(self) -> None:
        """Attempting to modify slot_index raises FrozenInstanceError."""
        slot = BatchSlotState(
            slot_index=0,
            request_id="req-1",
            slot_generation=1,
            is_active=True,
        )
        with pytest.raises(FrozenInstanceError):
            slot.slot_index = 5  # type: ignore[misc]

    def test_frozen_cannot_modify_request_id(self) -> None:
        """Attempting to modify request_id raises FrozenInstanceError."""
        slot = BatchSlotState(
            slot_index=0,
            request_id="req-1",
            slot_generation=1,
            is_active=True,
        )
        with pytest.raises(FrozenInstanceError):
            slot.request_id = "req-2"  # type: ignore[misc]

    def test_frozen_cannot_modify_is_active(self) -> None:
        """Attempting to modify is_active raises FrozenInstanceError."""
        slot = BatchSlotState(
            slot_index=0,
            request_id="req-1",
            slot_generation=1,
            is_active=True,
        )
        with pytest.raises(FrozenInstanceError):
            slot.is_active = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: DecodeMicrobatch correctly represents batch composition
# ---------------------------------------------------------------------------


class TestDecodeMicrobatch:
    """Tests for DecodeMicrobatch batch composition representation."""

    def test_creation_with_active_slots(self) -> None:
        """DecodeMicrobatch correctly stores batch composition."""
        slots = (
            BatchSlotState(slot_index=0, request_id="req-1", slot_generation=1, is_active=True),
            BatchSlotState(slot_index=1, request_id="req-2", slot_generation=1, is_active=True),
            BatchSlotState(slot_index=2, request_id=None, slot_generation=0, is_active=False),
            BatchSlotState(slot_index=3, request_id=None, slot_generation=0, is_active=False),
        )
        microbatch = DecodeMicrobatch(
            active_slot_count=2,
            max_batch_size=4,
            slot_states=slots,
            input_token_ids=(42, 99),
        )
        assert microbatch.active_slot_count == 2
        assert microbatch.max_batch_size == 4
        assert len(microbatch.slot_states) == 4
        assert microbatch.input_token_ids == (42, 99)

    def test_active_slot_count_matches_active_slots(self) -> None:
        """active_slot_count reflects the number of active slots."""
        slots = (
            BatchSlotState(slot_index=0, request_id="req-1", slot_generation=1, is_active=True),
            BatchSlotState(slot_index=1, request_id="req-2", slot_generation=2, is_active=True),
            BatchSlotState(slot_index=2, request_id="req-3", slot_generation=1, is_active=True),
        )
        microbatch = DecodeMicrobatch(
            active_slot_count=3,
            max_batch_size=8,
            slot_states=slots,
            input_token_ids=(10, 20, 30),
        )
        active_count = sum(1 for s in microbatch.slot_states if s.is_active)
        assert active_count == microbatch.active_slot_count

    def test_input_token_ids_length_matches_active_count(self) -> None:
        """input_token_ids length equals active_slot_count."""
        slots = (
            BatchSlotState(slot_index=0, request_id="req-1", slot_generation=1, is_active=True),
            BatchSlotState(slot_index=1, request_id=None, slot_generation=0, is_active=False),
        )
        microbatch = DecodeMicrobatch(
            active_slot_count=1,
            max_batch_size=2,
            slot_states=slots,
            input_token_ids=(55,),
        )
        assert len(microbatch.input_token_ids) == microbatch.active_slot_count

    def test_frozen_cannot_modify(self) -> None:
        """DecodeMicrobatch is frozen and cannot be modified."""
        slots = (
            BatchSlotState(slot_index=0, request_id="req-1", slot_generation=1, is_active=True),
        )
        microbatch = DecodeMicrobatch(
            active_slot_count=1,
            max_batch_size=4,
            slot_states=slots,
            input_token_ids=(42,),
        )
        with pytest.raises(FrozenInstanceError):
            microbatch.active_slot_count = 2  # type: ignore[misc]

    def test_empty_microbatch(self) -> None:
        """DecodeMicrobatch can represent an empty batch with no active slots."""
        microbatch = DecodeMicrobatch(
            active_slot_count=0,
            max_batch_size=8,
            slot_states=(),
            input_token_ids=(),
        )
        assert microbatch.active_slot_count == 0
        assert microbatch.max_batch_size == 8
        assert len(microbatch.slot_states) == 0
        assert len(microbatch.input_token_ids) == 0


# ---------------------------------------------------------------------------
# Test: TokenResultBatch is frozen and contains matching-length tuples
# ---------------------------------------------------------------------------


class TestTokenResultBatch:
    """Tests for TokenResultBatch immutability and tuple consistency."""

    def test_creation_with_results(self) -> None:
        """TokenResultBatch stores decode step results."""
        batch = TokenResultBatch(
            token_ids=(42, 99, 7),
            slot_indices=(0, 1, 2),
            slot_generations=(1, 1, 1),
        )
        assert batch.token_ids == (42, 99, 7)
        assert batch.slot_indices == (0, 1, 2)
        assert batch.slot_generations == (1, 1, 1)

    def test_frozen_cannot_modify_token_ids(self) -> None:
        """Attempting to modify token_ids raises FrozenInstanceError."""
        batch = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(0,),
            slot_generations=(1,),
        )
        with pytest.raises(FrozenInstanceError):
            batch.token_ids = (99,)  # type: ignore[misc]

    def test_frozen_cannot_modify_slot_indices(self) -> None:
        """Attempting to modify slot_indices raises FrozenInstanceError."""
        batch = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(0,),
            slot_generations=(1,),
        )
        with pytest.raises(FrozenInstanceError):
            batch.slot_indices = (5,)  # type: ignore[misc]

    def test_frozen_cannot_modify_slot_generations(self) -> None:
        """Attempting to modify slot_generations raises FrozenInstanceError."""
        batch = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(0,),
            slot_generations=(1,),
        )
        with pytest.raises(FrozenInstanceError):
            batch.slot_generations = (2,)  # type: ignore[misc]

    def test_matching_length_tuples(self) -> None:
        """All tuples in TokenResultBatch have the same length."""
        batch = TokenResultBatch(
            token_ids=(10, 20, 30, 40),
            slot_indices=(0, 1, 2, 3),
            slot_generations=(1, 2, 1, 3),
        )
        assert len(batch.token_ids) == len(batch.slot_indices)
        assert len(batch.token_ids) == len(batch.slot_generations)

    def test_single_result(self) -> None:
        """TokenResultBatch works with a single result."""
        batch = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(0,),
            slot_generations=(1,),
        )
        assert len(batch.token_ids) == 1
        assert batch.token_ids[0] == 42
        assert batch.slot_indices[0] == 0
        assert batch.slot_generations[0] == 1

    def test_empty_result_batch(self) -> None:
        """TokenResultBatch can represent an empty result set."""
        batch = TokenResultBatch(
            token_ids=(),
            slot_indices=(),
            slot_generations=(),
        )
        assert len(batch.token_ids) == 0
        assert len(batch.slot_indices) == 0
        assert len(batch.slot_generations) == 0

    def test_slot_generations_for_staleness_check(self) -> None:
        """slot_generations enables staleness detection across slot reuse."""
        # First batch: slot 0 has generation 1
        batch_1 = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(0,),
            slot_generations=(1,),
        )
        # Second batch: slot 0 reused with generation 2
        batch_2 = TokenResultBatch(
            token_ids=(99,),
            slot_indices=(0,),
            slot_generations=(2,),
        )
        # Different generations for same slot index
        assert batch_1.slot_generations[0] != batch_2.slot_generations[0]
