"""
Tests for ContinuousDecodeStep — the step-based continuous decode loop.

Covers:
- get_next_action returns "prefill" when a request needs prefilling
- get_next_action returns "decode" when requests are decode-ready
- get_next_action returns "idle" when no work is available
- report_decode_results correctly routes tokens to the engine
- A full multi-request session completes correctly
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
_DECODE_PATH = _ENGINE_DIR / "continuous_decode.py"


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
TokenResultBatch = _batching_mod.TokenResultBatch

# Load continuous_batching_engine
_engine_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching_engine", _ENGINE_PATH
)
ContinuousBatchingEngine = _engine_mod.ContinuousBatchingEngine

# Load continuous_decode
_decode_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_decode", _DECODE_PATH
)
ContinuousDecodeStep = _decode_mod.ContinuousDecodeStep
ContinuousDecodeAction = _decode_mod.ContinuousDecodeAction
ContinuousDecodeResult = _decode_mod.ContinuousDecodeResult
run_continuous_decode_steps = _decode_mod.run_continuous_decode_steps


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


def _submit_and_prefill(
    engine: ContinuousBatchingEngine,
    request_id: str,
    prompt_tokens: list[int],
    max_tokens: int = 5,
) -> None:
    """Submit a request and complete its prefill."""
    engine.submit_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling_config=_DEFAULT_SAMPLING,
    )
    engine.get_prefill_request()
    engine.mark_prefill_done(request_id)


# ---------------------------------------------------------------------------
# Test: get_next_action returns "prefill" when a request needs prefilling
# ---------------------------------------------------------------------------


class TestGetNextActionPrefill:
    """Verify get_next_action returns prefill actions correctly."""

    def test_returns_prefill_for_admitted_request(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[10, 20, 30],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()

        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-1"
        assert action.prefill_tokens == [10, 20, 30]
        assert action.decode_microbatch is None

    def test_prefill_has_priority_over_decode(self) -> None:
        """When both prefill and decode work exist, prefill wins."""
        engine = _make_engine()

        # First request is decode-ready
        _submit_and_prefill(engine, "req-1", [1, 2], max_tokens=10)

        # Second request is admitted (needs prefill)
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[3, 4, 5],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()

        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-2"


# ---------------------------------------------------------------------------
# Test: get_next_action returns "decode" when requests are decode-ready
# ---------------------------------------------------------------------------


class TestGetNextActionDecode:
    """Verify get_next_action returns decode actions correctly."""

    def test_returns_decode_when_requests_are_ready(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [10, 20], max_tokens=5)

        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()

        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        assert action.decode_microbatch.active_slot_count == 1
        assert action.prefill_request_id is None
        assert action.prefill_tokens is None

    def test_decode_microbatch_contains_all_ready_requests(self) -> None:
        engine = _make_engine(max_batch_size=4)
        _submit_and_prefill(engine, "req-1", [1], max_tokens=5)
        _submit_and_prefill(engine, "req-2", [2], max_tokens=5)
        _submit_and_prefill(engine, "req-3", [3], max_tokens=5)

        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()

        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        assert action.decode_microbatch.active_slot_count == 3


# ---------------------------------------------------------------------------
# Test: get_next_action returns "idle" when no work is available
# ---------------------------------------------------------------------------


class TestGetNextActionIdle:
    """Verify get_next_action returns idle when nothing to do."""

    def test_returns_idle_on_empty_engine(self) -> None:
        engine = _make_engine()
        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()

        assert action.action_type == "idle"
        assert action.prefill_request_id is None
        assert action.prefill_tokens is None
        assert action.decode_microbatch is None

    def test_returns_idle_after_all_requests_complete(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1], max_tokens=1)

        step = ContinuousDecodeStep(engine)

        # Get decode action
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None

        # Report result that completes the request
        slot_state = action.decode_microbatch.slot_states[0]
        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        step.report_decode_results(results)

        # Now should be idle
        action2 = step.get_next_action()
        assert action2.action_type == "idle"


# ---------------------------------------------------------------------------
# Test: report_decode_results correctly routes tokens to the engine
# ---------------------------------------------------------------------------


class TestReportDecodeResults:
    """Verify report_decode_results updates engine state correctly."""

    def test_routes_tokens_and_returns_completed(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1, 2], max_tokens=2)

        step = ContinuousDecodeStep(engine)

        # First decode step
        action = step.get_next_action()
        assert action.decode_microbatch is not None
        slot_state = action.decode_microbatch.slot_states[0]

        results = TokenResultBatch(
            token_ids=(100,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        completed = step.report_decode_results(results)
        assert completed == []

        # Second decode step — completes the request
        action2 = step.get_next_action()
        assert action2.decode_microbatch is not None

        results2 = TokenResultBatch(
            token_ids=(200,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation,),
        )
        completed2 = step.report_decode_results(results2)
        assert completed2 == ["req-1"]

    def test_tracks_total_tokens_generated(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1], max_tokens=3)

        step = ContinuousDecodeStep(engine)

        for token_id in [10, 20, 30]:
            action = step.get_next_action()
            assert action.decode_microbatch is not None
            slot_state = action.decode_microbatch.slot_states[0]

            results = TokenResultBatch(
                token_ids=(token_id,),
                slot_indices=(slot_state.slot_index,),
                slot_generations=(slot_state.slot_generation,),
            )
            step.report_decode_results(results)

        result = step.session_result
        assert result.total_tokens_generated == 3
        assert result.total_steps == 3
        assert result.completed_requests == ["req-1"]

    def test_stale_results_do_not_count_as_completion(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1], max_tokens=5)

        step = ContinuousDecodeStep(engine)
        action = step.get_next_action()
        assert action.decode_microbatch is not None
        slot_state = action.decode_microbatch.slot_states[0]

        # Report with wrong generation — should be discarded
        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_state.slot_index,),
            slot_generations=(slot_state.slot_generation + 999,),
        )
        completed = step.report_decode_results(results)
        assert completed == []
        # Request is still active
        assert engine.active_request_count == 1


# ---------------------------------------------------------------------------
# Test: report_prefill_done advances request to decode
# ---------------------------------------------------------------------------


class TestReportPrefillDone:
    """Verify report_prefill_done transitions request state."""

    def test_prefill_done_enables_decode(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)

        # Get prefill action
        action = step.get_next_action()
        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-1"

        # Report prefill done
        step.report_prefill_done("req-1")

        # Now decode should be available
        action2 = step.get_next_action()
        assert action2.action_type == "decode"
        assert action2.decode_microbatch is not None
        assert action2.decode_microbatch.active_slot_count == 1


# ---------------------------------------------------------------------------
# Test: report_cancellation removes request
# ---------------------------------------------------------------------------


class TestReportCancellation:
    """Verify report_cancellation removes request and tracks it."""

    def test_cancel_removes_from_engine(self) -> None:
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)
        step.report_cancellation("req-1")

        assert engine.has_work is False
        result = step.session_result
        assert result.cancelled_requests == ["req-1"]

    def test_cancel_decoding_request(self) -> None:
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1], max_tokens=10)

        step = ContinuousDecodeStep(engine)
        step.report_cancellation("req-1")

        assert engine.has_work is False
        action = step.get_next_action()
        assert action.action_type == "idle"


# ---------------------------------------------------------------------------
# Test: Full multi-request session completes correctly
# ---------------------------------------------------------------------------


class TestFullMultiRequestSession:
    """Verify a complete multi-request session with interleaved operations."""

    def test_two_requests_complete_sequentially(self) -> None:
        """Two requests: prefill both, then decode both to completion."""
        engine = _make_engine(max_batch_size=4)
        step = ContinuousDecodeStep(engine)

        # Submit two requests
        engine.submit_request(
            request_id="req-a",
            prompt_tokens=[1, 2],
            max_tokens=2,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-b",
            prompt_tokens=[3, 4],
            max_tokens=3,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Prefill req-a
        action = step.get_next_action()
        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-a"
        step.report_prefill_done("req-a")

        # Prefill req-b
        action = step.get_next_action()
        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-b"
        step.report_prefill_done("req-b")

        # Decode step 1: both requests in batch
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        assert action.decode_microbatch.active_slot_count == 2

        # Build results for both slots
        slot_a = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-a"
        )
        slot_b = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-b"
        )

        results = TokenResultBatch(
            token_ids=(100, 200),
            slot_indices=(slot_a.slot_index, slot_b.slot_index),
            slot_generations=(slot_a.slot_generation, slot_b.slot_generation),
        )
        completed = step.report_decode_results(results)
        assert completed == []

        # Decode step 2: req-a completes (max_tokens=2)
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None

        # Find slots again (req-a still in batch for this step)
        slot_a2 = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-a"
        )
        slot_b2 = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-b"
        )

        results2 = TokenResultBatch(
            token_ids=(101, 201),
            slot_indices=(slot_a2.slot_index, slot_b2.slot_index),
            slot_generations=(
                slot_a2.slot_generation,
                slot_b2.slot_generation,
            ),
        )
        completed2 = step.report_decode_results(results2)
        assert "req-a" in completed2

        # Decode step 3: only req-b remains, completes (max_tokens=3)
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        assert action.decode_microbatch.active_slot_count == 1

        slot_b3 = action.decode_microbatch.slot_states[0]
        assert slot_b3.request_id == "req-b"

        results3 = TokenResultBatch(
            token_ids=(202,),
            slot_indices=(slot_b3.slot_index,),
            slot_generations=(slot_b3.slot_generation,),
        )
        completed3 = step.report_decode_results(results3)
        assert "req-b" in completed3

        # Session complete
        assert step.has_work is False
        result = step.session_result
        assert result.total_tokens_generated == 5
        assert result.total_steps == 3
        assert set(result.completed_requests) == {"req-a", "req-b"}
        assert result.cancelled_requests == []

    def test_request_submitted_during_decode_joins_batch(self) -> None:
        """A request submitted mid-session joins after prefill."""
        engine = _make_engine(max_batch_size=4)
        step = ContinuousDecodeStep(engine)

        # Submit and prefill first request
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=3,
            sampling_config=_DEFAULT_SAMPLING,
        )
        action = step.get_next_action()
        assert action.action_type == "prefill"
        step.report_prefill_done("req-1")

        # Decode step 1 for req-1
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        slot_1 = action.decode_microbatch.slot_states[0]

        results = TokenResultBatch(
            token_ids=(10,),
            slot_indices=(slot_1.slot_index,),
            slot_generations=(slot_1.slot_generation,),
        )
        step.report_decode_results(results)

        # Submit second request mid-session
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[2, 3],
            max_tokens=2,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Next action should be prefill for req-2
        action = step.get_next_action()
        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-2"
        step.report_prefill_done("req-2")

        # Now both should be in decode batch
        action = step.get_next_action()
        assert action.action_type == "decode"
        assert action.decode_microbatch is not None
        assert action.decode_microbatch.active_slot_count == 2

    def test_mixed_completion_and_cancellation(self) -> None:
        """One request completes, another is cancelled."""
        engine = _make_engine(max_batch_size=4)
        step = ContinuousDecodeStep(engine)

        # Submit two requests
        engine.submit_request(
            request_id="req-complete",
            prompt_tokens=[1],
            max_tokens=1,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-cancel",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )

        # Prefill both
        action = step.get_next_action()
        step.report_prefill_done(action.prefill_request_id)  # type: ignore[arg-type]
        action = step.get_next_action()
        step.report_prefill_done(action.prefill_request_id)  # type: ignore[arg-type]

        # Decode step — complete req-complete
        action = step.get_next_action()
        assert action.decode_microbatch is not None

        slot_complete = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-complete"
        )
        slot_cancel = next(
            s for s in action.decode_microbatch.slot_states
            if s.request_id == "req-cancel"
        )

        results = TokenResultBatch(
            token_ids=(99, 88),
            slot_indices=(slot_complete.slot_index, slot_cancel.slot_index),
            slot_generations=(
                slot_complete.slot_generation,
                slot_cancel.slot_generation,
            ),
        )
        completed = step.report_decode_results(results)
        assert "req-complete" in completed

        # Cancel the remaining request
        step.report_cancellation("req-cancel")

        # Session done
        assert step.has_work is False
        result = step.session_result
        assert result.completed_requests == ["req-complete"]
        assert result.cancelled_requests == ["req-cancel"]
        assert result.total_tokens_generated == 2
        assert result.total_steps == 1


# ---------------------------------------------------------------------------
# Test: run_continuous_decode_steps driver
# ---------------------------------------------------------------------------


class TestRunContinuousDecodeSteps:
    """Verify the synchronous step driver function."""

    def test_auto_prefill_and_custom_decode(self) -> None:
        """Auto-prefill with custom decode callback completes session."""
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1, 2],
            max_tokens=2,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)
        decode_call_count = 0

        def execute_decode(microbatch: object) -> None:
            nonlocal decode_call_count
            decode_call_count += 1
            # Simulate generating a token for each active slot
            mb = microbatch  # type: ignore[assignment]
            token_ids = tuple(
                1000 + decode_call_count for _ in mb.slot_states
            )
            slot_indices = tuple(s.slot_index for s in mb.slot_states)
            slot_generations = tuple(
                s.slot_generation for s in mb.slot_states
            )
            results = TokenResultBatch(
                token_ids=token_ids,
                slot_indices=slot_indices,
                slot_generations=slot_generations,
            )
            step.report_decode_results(results)

        result = run_continuous_decode_steps(
            step=step,
            execute_decode=execute_decode,
        )

        assert result.total_tokens_generated == 2
        assert result.total_steps == 2
        assert result.completed_requests == ["req-1"]
        assert decode_call_count == 2

    def test_max_steps_limits_execution(self) -> None:
        """max_steps stops the loop before all requests complete."""
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=100,
            sampling_config=_DEFAULT_SAMPLING,
        )

        step = ContinuousDecodeStep(engine)

        def execute_decode(microbatch: object) -> None:
            mb = microbatch  # type: ignore[assignment]
            token_ids = tuple(42 for _ in mb.slot_states)
            slot_indices = tuple(s.slot_index for s in mb.slot_states)
            slot_generations = tuple(
                s.slot_generation for s in mb.slot_states
            )
            results = TokenResultBatch(
                token_ids=token_ids,
                slot_indices=slot_indices,
                slot_generations=slot_generations,
            )
            step.report_decode_results(results)

        result = run_continuous_decode_steps(
            step=step,
            execute_decode=execute_decode,
            max_steps=5,
        )

        assert result.total_steps == 5
        assert result.total_tokens_generated == 5
        assert result.completed_requests == []

    def test_no_decode_callback_raises(self) -> None:
        """Without execute_decode, decode action raises RuntimeError."""
        engine = _make_engine()
        _submit_and_prefill(engine, "req-1", [1], max_tokens=5)

        step = ContinuousDecodeStep(engine)

        with pytest.raises(RuntimeError, match="no execute_decode callback"):
            run_continuous_decode_steps(step=step)


# ---------------------------------------------------------------------------
# Test: ContinuousDecodeAction frozen dataclass
# ---------------------------------------------------------------------------


class TestContinuousDecodeAction:
    """Verify ContinuousDecodeAction is immutable and well-formed."""

    def test_idle_action_fields(self) -> None:
        action = ContinuousDecodeAction(action_type="idle")
        assert action.action_type == "idle"
        assert action.prefill_request_id is None
        assert action.prefill_tokens is None
        assert action.decode_microbatch is None

    def test_prefill_action_fields(self) -> None:
        action = ContinuousDecodeAction(
            action_type="prefill",
            prefill_request_id="req-1",
            prefill_tokens=[1, 2, 3],
        )
        assert action.action_type == "prefill"
        assert action.prefill_request_id == "req-1"
        assert action.prefill_tokens == [1, 2, 3]

    def test_action_is_frozen(self) -> None:
        action = ContinuousDecodeAction(action_type="idle")
        with pytest.raises(Exception):
            action.action_type = "decode"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: ContinuousDecodeResult frozen dataclass
# ---------------------------------------------------------------------------


class TestContinuousDecodeResult:
    """Verify ContinuousDecodeResult is immutable and well-formed."""

    def test_result_fields(self) -> None:
        result = ContinuousDecodeResult(
            total_tokens_generated=10,
            total_steps=5,
            completed_requests=["req-1", "req-2"],
            cancelled_requests=["req-3"],
        )
        assert result.total_tokens_generated == 10
        assert result.total_steps == 5
        assert result.completed_requests == ["req-1", "req-2"]
        assert result.cancelled_requests == ["req-3"]

    def test_result_is_frozen(self) -> None:
        result = ContinuousDecodeResult(
            total_tokens_generated=0,
            total_steps=0,
            completed_requests=[],
            cancelled_requests=[],
        )
        with pytest.raises(Exception):
            result.total_tokens_generated = 99  # type: ignore[misc]
