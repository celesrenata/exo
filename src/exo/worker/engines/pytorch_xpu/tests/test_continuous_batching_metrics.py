"""
Tests for ContinuousBatchingMetrics — extended metrics for continuous batching.

Covers:
- Metrics are computed correctly after a multi-step session
- Average microbatch size reflects actual batch sizes
- Pipeline occupancy is in [0.0, 1.0]
- Rates are computed correctly (admission rate, completion rate)
- Metrics at zero steps return zero for derived fields
- Cancellation count is tracked
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
ContinuousBatchingMetrics = _engine_mod.ContinuousBatchingMetrics


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
    prompt_tokens: list[int] | None = None,
    max_tokens: int = 10,
) -> tuple[int, int]:
    """Submit a request, prefill it, and return (slot_index, slot_generation)."""
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3]
    engine.submit_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling_config=_DEFAULT_SAMPLING,
    )
    engine.get_prefill_request()
    engine.mark_prefill_done(request_id)

    microbatch = engine.get_decode_microbatch()
    assert microbatch is not None
    for slot in microbatch.slot_states:
        if slot.request_id == request_id:
            return slot.slot_index, slot.slot_generation
    raise AssertionError(f"Request {request_id} not found in microbatch")


# ---------------------------------------------------------------------------
# Test: Metrics at zero steps
# ---------------------------------------------------------------------------


class TestMetricsZeroSteps:
    """Verify metrics return sensible defaults when no decode steps have run."""

    def test_fresh_engine_metrics(self) -> None:
        engine = _make_engine()
        metrics = engine.get_extended_metrics()

        assert metrics.active_requests == 0
        assert metrics.admission_queue_size == 0
        assert metrics.decode_ready_count == 0
        assert metrics.completed_count == 0
        assert metrics.cancelled_count == 0
        assert metrics.total_decode_steps == 0
        assert metrics.total_tokens_generated == 0
        assert metrics.total_admitted == 0
        assert metrics.average_microbatch_size == 0.0
        assert metrics.pipeline_occupancy == 0.0
        assert metrics.admission_rate_per_step == 0.0
        assert metrics.completion_rate_per_step == 0.0

    def test_metrics_after_submit_before_decode(self) -> None:
        """Metrics reflect admitted requests before any decode step."""
        engine = _make_engine()
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1, 2],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        metrics = engine.get_extended_metrics()

        assert metrics.active_requests == 1
        assert metrics.admission_queue_size == 1
        assert metrics.total_admitted == 1
        assert metrics.total_decode_steps == 0
        assert metrics.average_microbatch_size == 0.0

    def test_metrics_is_frozen_dataclass(self) -> None:
        """ContinuousBatchingMetrics is immutable."""
        engine = _make_engine()
        metrics = engine.get_extended_metrics()
        with pytest.raises(Exception):  # noqa: B017
            metrics.active_requests = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: Metrics after multi-step session
# ---------------------------------------------------------------------------


class TestMetricsMultiStep:
    """Verify metrics are computed correctly after a multi-step session."""

    def test_single_request_full_lifecycle(self) -> None:
        """One request, 3 tokens generated, then completed."""
        engine = _make_engine(max_batch_size=4)
        slot_idx, slot_gen = _submit_and_prefill(engine, "req-1", max_tokens=3)

        # Generate 3 tokens
        for token_id in [10, 20, 30]:
            results = TokenResultBatch(
                token_ids=(token_id,),
                slot_indices=(slot_idx,),
                slot_generations=(slot_gen,),
            )
            engine.report_token_results(results)

        metrics = engine.get_extended_metrics()

        assert metrics.total_decode_steps == 3
        assert metrics.total_tokens_generated == 3
        assert metrics.total_admitted == 1
        assert metrics.completed_count == 1
        assert metrics.active_requests == 0
        assert metrics.average_microbatch_size == 1.0  # 1 token per step
        assert metrics.admission_rate_per_step == pytest.approx(1.0 / 3.0)
        assert metrics.completion_rate_per_step == pytest.approx(1.0 / 3.0)

    def test_two_requests_batched_together(self) -> None:
        """Two requests decoded together produce batch size 2."""
        engine = _make_engine(max_batch_size=4)

        # Submit and prefill both
        engine.submit_request(
            request_id="req-a",
            prompt_tokens=[1],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.submit_request(
            request_id="req-b",
            prompt_tokens=[2],
            max_tokens=5,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-a")
        engine.get_prefill_request()
        engine.mark_prefill_done("req-b")

        # Get microbatch to find slot info
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 2

        slot_a = next(
            s for s in microbatch.slot_states if s.request_id == "req-a"
        )
        slot_b = next(
            s for s in microbatch.slot_states if s.request_id == "req-b"
        )

        # Report tokens for both in one batch (simulates one decode step)
        results = TokenResultBatch(
            token_ids=(100, 200),
            slot_indices=(slot_a.slot_index, slot_b.slot_index),
            slot_generations=(slot_a.slot_generation, slot_b.slot_generation),
        )
        engine.report_token_results(results)

        metrics = engine.get_extended_metrics()

        # One decode step with 2 tokens
        assert metrics.total_decode_steps == 1
        assert metrics.total_tokens_generated == 2
        assert metrics.average_microbatch_size == 2.0
        assert metrics.total_admitted == 2
        assert metrics.admission_rate_per_step == 2.0

    def test_multiple_steps_varying_batch_size(self) -> None:
        """Average microbatch size reflects varying batch sizes across steps."""
        engine = _make_engine(max_batch_size=4)

        # Submit and prefill first request
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_1 = microbatch.slot_states[0]

        # Step 1: batch size 1
        results = TokenResultBatch(
            token_ids=(10,),
            slot_indices=(slot_1.slot_index,),
            slot_generations=(slot_1.slot_generation,),
        )
        engine.report_token_results(results)

        # Submit and prefill second request
        engine.submit_request(
            request_id="req-2",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-2")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_2 = next(
            s for s in microbatch.slot_states if s.request_id == "req-2"
        )

        # Step 2: batch size 2
        results = TokenResultBatch(
            token_ids=(20, 30),
            slot_indices=(slot_1.slot_index, slot_2.slot_index),
            slot_generations=(slot_1.slot_generation, slot_2.slot_generation),
        )
        engine.report_token_results(results)

        # Step 3: batch size 2
        results = TokenResultBatch(
            token_ids=(40, 50),
            slot_indices=(slot_1.slot_index, slot_2.slot_index),
            slot_generations=(slot_1.slot_generation, slot_2.slot_generation),
        )
        engine.report_token_results(results)

        metrics = engine.get_extended_metrics()

        # 3 steps: sizes 1, 2, 2 → total 5, average 5/3
        assert metrics.total_decode_steps == 3
        assert metrics.total_tokens_generated == 5
        assert metrics.average_microbatch_size == pytest.approx(5.0 / 3.0)


# ---------------------------------------------------------------------------
# Test: Pipeline occupancy is in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestPipelineOccupancy:
    """Verify pipeline occupancy is bounded and reflects decode-ready slots."""

    def test_occupancy_zero_when_no_decode_ready(self) -> None:
        engine = _make_engine(max_batch_size=4)
        metrics = engine.get_extended_metrics()
        assert metrics.pipeline_occupancy == 0.0

    def test_occupancy_quarter_with_one_of_four(self) -> None:
        engine = _make_engine(max_batch_size=4)
        _submit_and_prefill(engine, "req-1")

        metrics = engine.get_extended_metrics()
        assert metrics.pipeline_occupancy == pytest.approx(0.25)

    def test_occupancy_full_at_max_batch_size(self) -> None:
        engine = _make_engine(max_batch_size=3)

        for i in range(3):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i],
                max_tokens=10,
                sampling_config=_DEFAULT_SAMPLING,
            )
            engine.get_prefill_request()
            engine.mark_prefill_done(f"req-{i}")

        metrics = engine.get_extended_metrics()
        assert metrics.pipeline_occupancy == pytest.approx(1.0)

    def test_occupancy_bounded_zero_to_one(self) -> None:
        """Occupancy never exceeds 1.0 regardless of state."""
        engine = _make_engine(max_batch_size=2)

        # Fill to capacity
        for i in range(2):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i],
                max_tokens=10,
                sampling_config=_DEFAULT_SAMPLING,
            )
            engine.get_prefill_request()
            engine.mark_prefill_done(f"req-{i}")

        metrics = engine.get_extended_metrics()
        assert 0.0 <= metrics.pipeline_occupancy <= 1.0


# ---------------------------------------------------------------------------
# Test: Rates are computed correctly
# ---------------------------------------------------------------------------


class TestRates:
    """Verify admission and completion rates are computed correctly."""

    def test_admission_rate_multiple_requests(self) -> None:
        """Admission rate = total_admitted / total_decode_steps."""
        engine = _make_engine(max_batch_size=4)

        # Submit 4 requests, prefill all
        for i in range(4):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i],
                max_tokens=10,
                sampling_config=_DEFAULT_SAMPLING,
            )
            engine.get_prefill_request()
            engine.mark_prefill_done(f"req-{i}")

        # Get microbatch for slot info
        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Run 2 decode steps
        for _ in range(2):
            token_ids = tuple(range(4))
            slot_indices = tuple(s.slot_index for s in microbatch.slot_states)
            slot_gens = tuple(
                s.slot_generation for s in microbatch.slot_states
            )
            results = TokenResultBatch(
                token_ids=token_ids,
                slot_indices=slot_indices,
                slot_generations=slot_gens,
            )
            engine.report_token_results(results)

        metrics = engine.get_extended_metrics()

        # 4 admitted / 2 steps = 2.0
        assert metrics.admission_rate_per_step == pytest.approx(2.0)

    def test_completion_rate(self) -> None:
        """Completion rate = completed_count / total_decode_steps."""
        engine = _make_engine(max_batch_size=4)

        # Submit 2 requests with max_tokens=2
        for i in range(2):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i],
                max_tokens=2,
                sampling_config=_DEFAULT_SAMPLING,
            )
            engine.get_prefill_request()
            engine.mark_prefill_done(f"req-{i}")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Step 1: both get a token
        results = TokenResultBatch(
            token_ids=tuple(range(2)),
            slot_indices=tuple(s.slot_index for s in microbatch.slot_states),
            slot_generations=tuple(
                s.slot_generation for s in microbatch.slot_states
            ),
        )
        engine.report_token_results(results)

        # Step 2: both complete (max_tokens=2)
        results = TokenResultBatch(
            token_ids=tuple(range(2)),
            slot_indices=tuple(s.slot_index for s in microbatch.slot_states),
            slot_generations=tuple(
                s.slot_generation for s in microbatch.slot_states
            ),
        )
        engine.report_token_results(results)

        metrics = engine.get_extended_metrics()

        # 2 completed / 2 steps = 1.0
        assert metrics.completion_rate_per_step == pytest.approx(1.0)
        assert metrics.completed_count == 2

    def test_cancellation_tracked(self) -> None:
        """Cancelled count is reflected in metrics."""
        engine = _make_engine(max_batch_size=4)

        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=_DEFAULT_SAMPLING,
        )
        engine.cancel_request("req-1")

        metrics = engine.get_extended_metrics()
        assert metrics.cancelled_count == 1
        assert metrics.total_admitted == 1
        assert metrics.active_requests == 0

    def test_stale_tokens_not_counted(self) -> None:
        """Stale token results (wrong generation) do not increment counters."""
        engine = _make_engine(max_batch_size=1)
        slot_idx, slot_gen = _submit_and_prefill(engine, "req-1", max_tokens=5)

        # Report with wrong generation
        results = TokenResultBatch(
            token_ids=(42,),
            slot_indices=(slot_idx,),
            slot_generations=(slot_gen + 999,),
        )
        engine.report_token_results(results)

        metrics = engine.get_extended_metrics()
        assert metrics.total_decode_steps == 0
        assert metrics.total_tokens_generated == 0
        assert metrics.average_microbatch_size == 0.0
