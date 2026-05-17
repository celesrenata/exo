"""
Tests for ContinuousBatchingEngine — the coordination layer for continuous batching.

Covers:
- Submit request → get_prefill_request returns it
- After mark_prefill_done → get_decode_microbatch includes it
- report_token_results updates state and detects completion
- cancel_request removes from all queues
- Multiple concurrent requests form a batch
- has_work reflects pending work
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
_ENGINE_PATH = _ENGINE_DIR / "continuous_batching_engine.py"


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

# Also mock pydantic if needed — sampling uses it
# pydantic should be available since it's a core dependency

# Load sampling first (dependency of continuous_batching)
_sampling_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.sampling", _SAMPLING_PATH
)
SamplingConfiguration = _sampling_mod.SamplingConfiguration

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
TokenResultBatch = _batching_mod.TokenResultBatch

# Load continuous_batching_engine
_engine_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching_engine", _ENGINE_PATH
)
ContinuousBatchingEngine = _engine_mod.ContinuousBatchingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLING = SamplingConfiguration(do_sample=False)


def _make_engine(
    max_batch_size: int = 4,
    num_layers: int = 8,
    world_size: int = 4,
) -> ContinuousBatchingEngine:
    """Create an engine with default test parameters."""
    return ContinuousBatchingEngine(
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Test: Submit request → get_prefill_request returns it
# ---------------------------------------------------------------------------


class TestSubmitAndPrefill:
    """Verify that submitted requests appear in the prefill queue."""

    def test_submit_request_appears_in_prefill(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )

        result = engine.get_prefill_request()
        assert result is not None
        request_id, prompt_tokens = result
        assert request_id == "req-1"
        assert prompt_tokens == [1, 2, 3]

    def test_get_prefill_request_returns_none_when_empty(self) -> None:
        engine = _make_engine()
        assert engine.get_prefill_request() is None

    def test_submit_duplicate_request_raises(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        with pytest.raises(ValueError, match="already submitted"):
            engine.submit_request(
                request_id="req-1",
                prompt_tokens=[2],
                max_tokens=5,
                sampling_config=_DEFAULT_SAMPLING,
            )

    def test_submit_with_invalid_max_tokens_raises(self) -> None:
        engine = _make_engine()
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            engine.submit_request(
                request_id="req-1",
                prompt_tokens=[1],
                max_tokens=0,
                sampling_config=_DEFAULT_SAMPLING,
            )

    def test_only_one_prefill_at_a_time(self) -> None:
        """Second request stays queued while first is prefilling."""
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1, 2],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[3, 4],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # First call gets req-1
        result = engine.get_prefill_request()
        assert result is not None
        assert result[0] == "req-1"

        # Second call still returns req-1 (active prefill)
        result2 = engine.get_prefill_request()
        assert result2 is not None
        assert result2[0] == "req-1"


# ---------------------------------------------------------------------------
# Test: After mark_prefill_done → get_decode_microbatch includes it
# ---------------------------------------------------------------------------


class TestPrefillToDecode:
    """Verify that completing prefill moves requests to decode batch."""

    def test_mark_prefill_done_moves_to_decode(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[10, 20, 30],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Start prefill
        result = engine.get_prefill_request()
        assert result is not None

        # Complete prefill
        engine.mark_prefill_done("req-1")

        # Now decode microbatch should include it
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 1
        assert microbatch.slot_states[0].request_id == "req-1"
        assert microbatch.slot_states[0].is_active is True

    def test_mark_prefill_done_wrong_request_raises(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()

        with pytest.raises(ValueError, match="not the active prefill"):
            engine.mark_prefill_done("req-wrong")

    def test_get_decode_microbatch_returns_none_when_empty(self) -> None:
        engine = _make_engine()
        assert engine.get_decode_microbatch() is None

    def test_after_prefill_done_next_prefill_available(self) -> None:
        """After completing one prefill, the next request can start."""
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[2],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Prefill req-1
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        # Now req-2 should be available for prefill
        result = engine.get_prefill_request()
        assert result is not None
        assert result[0] == "req-2"


# ---------------------------------------------------------------------------
# Test: report_token_results updates state and detects completion
# ---------------------------------------------------------------------------


class TestTokenResults:
    """Verify token result reporting and completion detection."""

    def _setup_decoding_request(
        self, engine: ContinuousBatchingEngine, request_id: str = "req-1"
    ) -> tuple[int, int]:
        """Submit, prefill, and move a request to decode. Returns (slot, gen)."""
        engine.submit_request(
            request_id=request_id,
            prompt_tokens=[1, 2, 3],
            max_tokens=3,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done(request_id)

        # Get the slot info
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_state = microbatch.slot_states[0]
        return slot_state.slot_index, slot_state.slot_generation

    def test_report_token_updates_state(self) -> None:
        engine = _make_engine()
        slot_index, slot_gen = self._setup_decoding_request(engine)

        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_index,),
            slot_generations=(slot_gen,),
        )
        completed = engine.report_token_results(results)
        assert completed == []
        assert engine.active_request_count == 1

    def test_report_token_detects_completion(self) -> None:
        engine = _make_engine()
        slot_index, slot_gen = self._setup_decoding_request(engine)

        # Generate 3 tokens (max_tokens=3)
        for token_id in [10, 20, 30]:
            results = TokenResultBatch(
                token_ids=(token_id,),
                slot_indices=(slot_index,),
                slot_generations=(slot_gen,),
            )
            completed = engine.report_token_results(results)

        # Last report should detect completion
        assert completed == ["req-1"]
        assert engine.active_request_count == 0
        assert engine.has_work is False

    def test_report_token_invokes_callback(self) -> None:
        engine = _make_engine()
        received_tokens: list[tuple[str, int]] = []

        def callback(rid: str, tid: int) -> None:
            received_tokens.append((rid, tid))

        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
            on_token=callback,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_state = microbatch.slot_states[0]

        results = TokenResultBatch(
            token_ids=(99,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        engine.report_token_results(results)

        assert received_tokens == [("req-1", 99)]

    def test_stale_generation_discarded(self) -> None:
        """Token results with wrong generation are silently discarded."""
        engine = _make_engine()
        slot_index, slot_gen = self._setup_decoding_request(engine)

        # Report with wrong generation
        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_index,),
            slot_generations=(slot_gen + 999,),  # stale
        )
        completed = engine.report_token_results(results)
        assert completed == []


# ---------------------------------------------------------------------------
# Test: cancel_request removes from all queues
# ---------------------------------------------------------------------------


class TestCancellation:
    """Verify cancellation removes requests from all states."""

    def test_cancel_admitted_request(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        assert engine.has_work is True

        engine.cancel_request("req-1")
        assert engine.has_work is False
        assert engine.active_request_count == 0
        assert engine.get_prefill_request() is None

    def test_cancel_prefilling_request(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()  # starts prefill

        engine.cancel_request("req-1")
        assert engine.has_work is False
        assert engine.get_prefill_request() is None

    def test_cancel_decoding_request(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        assert engine.decode_batch_size == 1
        engine.cancel_request("req-1")
        assert engine.decode_batch_size == 0
        assert engine.has_work is False
        assert engine.get_decode_microbatch() is None

    def test_cancel_unknown_request_is_noop(self) -> None:
        engine = _make_engine()
        # Should not raise
        engine.cancel_request("nonexistent")

    def test_cancel_already_completed_is_noop(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=1,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_state = microbatch.slot_states[0]

        # Complete the request
        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        engine.report_token_results(results)

        # Cancel after completion — should not raise
        engine.cancel_request("req-1")


# ---------------------------------------------------------------------------
# Test: Multiple concurrent requests form a batch
# ---------------------------------------------------------------------------


class TestMultipleConcurrentRequests:
    """Verify multiple requests can be batched together for decode."""

    def test_multiple_requests_in_decode_batch(self) -> None:
        engine = _make_engine(max_batch_size=4)

        # Submit 3 requests
        for i in range(3):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i * 10 + 1, i * 10 + 2],
                max_tokens=5,
                sampling_config=_DEFAULT_SAMPLING,
            )

        # Prefill all three sequentially
        for i in range(3):
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        # All three should be in the decode batch
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 3

        request_ids_in_batch = {
            s.request_id for s in microbatch.slot_states
        }
        assert request_ids_in_batch == {"req-0", "req-1", "req-2"}

    def test_new_request_joins_after_prefill(self) -> None:
        """A request submitted during decode joins after its prefill."""
        engine = _make_engine(max_batch_size=4)

        # First request goes through full cycle
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        # Decode batch has 1 request
        assert engine.decode_batch_size == 1

        # Submit second request while first is decoding
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Prefill second request
        result = engine.get_prefill_request()
        assert result is not None
        assert result[0] == "req-2"
        engine.mark_prefill_done("req-2")

        # Now both should be in decode batch
        assert engine.decode_batch_size == 2
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 2

    def test_batch_shrinks_on_completion(self) -> None:
        """Completed requests are removed from the decode batch."""
        engine = _make_engine(max_batch_size=4)

        # Submit and prefill 2 requests with different max_tokens
        engine.submit_request(
            request_id="req-short",
            prompt_tokens=[1],
            max_tokens=1,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-long",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )

        engine.get_prefill_request()
        engine.mark_prefill_done("req-short")
        engine.get_prefill_request()
        engine.mark_prefill_done("req-long")

        assert engine.decode_batch_size == 2

        # Get microbatch and report results
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Find slot info for each request
        short_slot = next(
            s for s in microbatch.slot_states if s.request_id == "req-short"
        )
        long_slot = next(
            s for s in microbatch.slot_states if s.request_id == "req-long"
        )

        # Report tokens for both — short completes (max_tokens=1)
        results = TokenResultBatch(
            token_ids=(100, 200),
            slot_indices=(short_slot.slot_index, long_slot.slot_index),
            slot_generations=(
                short_slot.slot_generation,
                long_slot.slot_generation,
            ),
        )
        completed = engine.report_token_results(results)
        assert "req-short" in completed
        assert "req-long" not in completed

        # Decode batch should now have only req-long
        assert engine.decode_batch_size == 1
        assert engine.active_request_count == 1


# ---------------------------------------------------------------------------
# Test: has_work reflects pending work
# ---------------------------------------------------------------------------


class TestHasWork:
    """Verify has_work property accurately reflects engine state."""

    def test_empty_engine_has_no_work(self) -> None:
        engine = _make_engine()
        assert engine.has_work is False

    def test_admitted_request_has_work(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        assert engine.has_work is True

    def test_prefilling_request_has_work(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        assert engine.has_work is True

    def test_decoding_request_has_work(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")
        assert engine.has_work is True

    def test_all_completed_no_work(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=1,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_state = microbatch.slot_states[0]

        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        engine.report_token_results(results)

        assert engine.has_work is False

    def test_cancelled_request_no_work(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.cancel_request("req-1")
        assert engine.has_work is False


# ---------------------------------------------------------------------------
# Test: Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Verify constructor rejects invalid parameters."""

    def test_zero_max_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            ContinuousBatchingEngine(
                max_batch_size=0, num_layers=8, world_size=4
            )

    def test_negative_num_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be positive"):
            ContinuousBatchingEngine(
                max_batch_size=4, num_layers=-1, world_size=4
            )

    def test_zero_world_size_raises(self) -> None:
        with pytest.raises(ValueError, match="world_size must be positive"):
            ContinuousBatchingEngine(
                max_batch_size=4, num_layers=8, world_size=0
            )
