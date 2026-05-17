"""
Tests for per-request microbatch sampling.

Verifies:
- Greedy sampling returns argmax for each request
- Different requests can have different sampling configs in the same batch
- Results contain correct slot indices and generations
- Empty microbatch returns empty results
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Torch mock — tests mock the logits tensor since torch is not available
# in the CI/dev environment. The sampling functions are tested via mocking.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent


def _create_torch_mock() -> types.ModuleType:
    """Create a minimal torch mock that supports tensor operations for testing."""
    torch_mock = types.ModuleType("torch")

    class MockTensor:
        """Mock tensor that supports argmax, topk, softmax, multinomial."""

        def __init__(self, data: list[list[float]] | list[float]) -> None:
            if not data:
                self._data: list[list[float]] = []
                self._ndim = 2
            elif isinstance(data[0], list):
                self._data = data  # type: ignore[assignment]
                self._ndim = 2
            else:
                self._data = [data]  # type: ignore[list-item]
                self._ndim = 1

        @property
        def shape(self) -> tuple[int, ...]:
            if self._ndim == 2:
                if not self._data:
                    return (0, 0)
                return (len(self._data), len(self._data[0]))
            return (len(self._data[0]),)

        @property
        def ndim(self) -> int:
            return self._ndim

        @property
        def device(self) -> str:
            return "cpu"

        def __getitem__(self, idx: Any) -> MockTensor:
            if isinstance(idx, int):
                return MockTensor(self._data[idx])
            return self

        def __truediv__(self, other: float | int) -> MockTensor:
            result = []
            for row in self._data:
                result.append([v / other for v in row])
            return MockTensor(result)

        def argmax(self, dim: int = -1) -> MockTensor:
            if self._ndim == 1 or (self._ndim == 2 and len(self._data) == 1):
                row = self._data[0] if self._ndim == 2 else self._data[0]
                max_idx = row.index(max(row))
                return MockTensor([[float(max_idx)]])
            # For 2D, argmax along last dim
            results = []
            for row in self._data:
                results.append(float(row.index(max(row))))
            return MockTensor([results])

        def item(self) -> float:
            if self._ndim == 2:
                return self._data[0][0]
            return self._data[0][0]

        def size(self, dim: int) -> int:
            shape = self.shape
            return shape[dim]

        def any(self) -> bool:
            for row in self._data:
                for val in row:
                    if val:
                        return True
            return False

        def unsqueeze(self, dim: int) -> MockTensor:
            return self

        def squeeze(self) -> MockTensor:
            return self

    def zeros(*args: Any, **kwargs: Any) -> MockTensor:
        if len(args) == 2:
            if args[0] == 0:
                return MockTensor([])
            return MockTensor([[0.0] * args[1] for _ in range(args[0])])
        elif len(args) == 1:
            return MockTensor([[0.0] * args[0]])
        return MockTensor([[0.0]])

    def empty(*args: Any, **kwargs: Any) -> MockTensor:
        return zeros(*args)

    def randn(*args: Any, **kwargs: Any) -> MockTensor:
        import random
        if len(args) == 2:
            return MockTensor(
                [[random.gauss(0, 1) for _ in range(args[1])] for _ in range(args[0])]
            )
        elif len(args) == 1:
            return MockTensor([[random.gauss(0, 1) for _ in range(args[0])]])
        return MockTensor([[random.gauss(0, 1)]])

    def topk(tensor: MockTensor, k: int) -> tuple[MockTensor, MockTensor]:
        row = tensor._data[0]
        indexed = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:k]
        values = [v for _, v in indexed]
        indices = [float(i) for i, _ in indexed]
        return MockTensor([values]), MockTensor([indices])

    def softmax(tensor: MockTensor, dim: int = -1) -> MockTensor:
        import math
        row = tensor._data[0]
        max_val = max(row)
        exp_vals = [math.exp(v - max_val) for v in row]
        total = sum(exp_vals)
        return MockTensor([[v / total for v in exp_vals]])

    def multinomial(tensor: MockTensor, num_samples: int = 1) -> MockTensor:
        import random
        row = tensor._data[0]
        total = sum(row)
        if total == 0:
            return MockTensor([[0.0]])
        normalized = [v / total for v in row]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(normalized):
            cumulative += p
            if r <= cumulative:
                return MockTensor([[float(i)]])
        return MockTensor([[float(len(row) - 1)]])

    def isnan(tensor: MockTensor) -> MockTensor:
        import math
        results = []
        for row in tensor._data:
            results.append([1.0 if math.isnan(v) else 0.0 for v in row])
        return MockTensor(results)

    def isinf(tensor: MockTensor) -> MockTensor:
        import math
        results = []
        for row in tensor._data:
            results.append([1.0 if math.isinf(v) else 0.0 for v in row])
        return MockTensor(results)

    torch_mock.Tensor = MockTensor  # type: ignore[attr-defined]
    torch_mock.zeros = zeros  # type: ignore[attr-defined]
    torch_mock.empty = empty  # type: ignore[attr-defined]
    torch_mock.randn = randn  # type: ignore[attr-defined]
    torch_mock.topk = topk  # type: ignore[attr-defined]
    torch_mock.softmax = softmax  # type: ignore[attr-defined]
    torch_mock.multinomial = multinomial  # type: ignore[attr-defined]
    torch_mock.isnan = isnan  # type: ignore[attr-defined]
    torch_mock.isinf = isinf  # type: ignore[attr-defined]
    torch_mock.long = "long"  # type: ignore[attr-defined]

    return torch_mock


# Install torch mock before importing our modules
_torch_mock = _create_torch_mock()
if "torch" not in sys.modules:
    sys.modules["torch"] = _torch_mock


# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------

_SAMPLING_PATH = _ENGINE_DIR / "sampling.py"
_BATCHING_PATH = _ENGINE_DIR / "continuous_batching.py"
_ENGINE_PATH = _ENGINE_DIR / "continuous_batching_engine.py"
_MICROBATCH_SAMPLING_PATH = _ENGINE_DIR / "microbatch_sampling.py"


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


_sampling_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.sampling", _SAMPLING_PATH
)
SamplingConfiguration = _sampling_mod.SamplingConfiguration

_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
DecodeMicrobatch = _batching_mod.DecodeMicrobatch
BatchSlotState = _batching_mod.BatchSlotState
TokenResultBatch = _batching_mod.TokenResultBatch

_engine_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching_engine", _ENGINE_PATH
)
ContinuousBatchingEngine = _engine_mod.ContinuousBatchingEngine

_microbatch_sampling_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.microbatch_sampling",
    _MICROBATCH_SAMPLING_PATH,
)
sample_microbatch = _microbatch_sampling_mod.sample_microbatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    max_batch_size: int = 4,
    num_layers: int = 8,
    world_size: int = 4,
) -> Any:
    """Create an engine with default test parameters."""
    return ContinuousBatchingEngine(
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        world_size=world_size,
    )


def _setup_decoding_request(
    engine: Any,
    request_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
    sampling_config: Any,
) -> None:
    """Submit, prefill, and move a request to decode state."""
    engine.submit_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        sampling_config=sampling_config,
    )
    engine.get_prefill_request()
    engine.mark_prefill_done(request_id)


def _make_logits(rows: int, vocab_size: int) -> Any:
    """Create a mock logits tensor of zeros."""
    return _torch_mock.zeros(rows, vocab_size)


def _set_logit(logits: Any, row: int, col: int, value: float) -> None:
    """Set a specific logit value in the mock tensor."""
    logits._data[row][col] = value


# ---------------------------------------------------------------------------
# Test: Greedy sampling returns argmax for each request
# ---------------------------------------------------------------------------


class TestGreedySampling:
    """Verify greedy sampling returns argmax for each request in microbatch."""

    def test_single_request_greedy_returns_argmax(self) -> None:
        """A single greedy request returns the token with highest logit."""
        engine = _make_engine()
        greedy_config = SamplingConfiguration(do_sample=False)

        _setup_decoding_request(
            engine, "req-1", [1, 2, 3], max_tokens=10, sampling_config=greedy_config
        )

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Logits where token 42 has the highest value
        logits = _make_logits(1, 100)
        _set_logit(logits, 0, 42, 10.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert result.token_ids == (42,)

    def test_multiple_greedy_requests_return_argmax(self) -> None:
        """Multiple greedy requests each return their own argmax."""
        engine = _make_engine(max_batch_size=4)
        greedy_config = SamplingConfiguration(do_sample=False)

        # Submit and prefill 3 requests
        for i in range(3):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i + 1],
                max_tokens=10,
                sampling_config=greedy_config,
            )

        # Prefill all three
        for i in range(3):
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 3

        # Each row has a different argmax
        logits = _make_logits(3, 50)
        _set_logit(logits, 0, 10, 5.0)  # first slot → token 10
        _set_logit(logits, 1, 25, 8.0)  # second slot → token 25
        _set_logit(logits, 2, 49, 3.0)  # third slot → token 49

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert len(result.token_ids) == 3
        assert 10 in result.token_ids
        assert 25 in result.token_ids
        assert 49 in result.token_ids

    def test_greedy_with_zero_temperature_returns_argmax(self) -> None:
        """Temperature=0 triggers greedy route and returns argmax."""
        engine = _make_engine()
        config = SamplingConfiguration(do_sample=True, temperature=0.0)

        _setup_decoding_request(
            engine, "req-1", [1], max_tokens=10, sampling_config=config
        )

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        logits = _make_logits(1, 200)
        _set_logit(logits, 0, 150, 100.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert result.token_ids == (150,)


# ---------------------------------------------------------------------------
# Test: Different requests can have different sampling configs
# ---------------------------------------------------------------------------


class TestMixedSamplingConfigs:
    """Verify different sampling configs work in the same microbatch."""

    def test_greedy_and_top_k_in_same_batch(self) -> None:
        """One greedy request and one top-k request in the same microbatch."""
        engine = _make_engine(max_batch_size=4)

        greedy_config = SamplingConfiguration(do_sample=False)
        top_k_config = SamplingConfiguration(
            do_sample=True, top_k=5, temperature=1.0
        )

        # Submit both
        engine.submit_request(
            request_id="greedy-req",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=greedy_config,
        )
        engine.submit_request(
            request_id="topk-req",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=top_k_config,
        )

        # Prefill both
        result = engine.get_prefill_request()
        assert result is not None
        engine.mark_prefill_done(result[0])
        result = engine.get_prefill_request()
        assert result is not None
        engine.mark_prefill_done(result[0])

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        assert microbatch.active_slot_count == 2

        # Build logits: greedy should pick token 99, top-k picks from top 5
        logits = _make_logits(2, 100)

        # Find which batch index corresponds to which request
        active_slots = [s for s in microbatch.slot_states if s.is_active]
        greedy_idx = next(
            i for i, s in enumerate(active_slots) if s.request_id == "greedy-req"
        )
        topk_idx = next(
            i for i, s in enumerate(active_slots) if s.request_id == "topk-req"
        )

        # Greedy request: token 99 is clearly the max
        _set_logit(logits, greedy_idx, 99, 100.0)

        # Top-k request: make top-5 tokens have high logits
        _set_logit(logits, topk_idx, 10, 5.0)
        _set_logit(logits, topk_idx, 20, 4.0)
        _set_logit(logits, topk_idx, 30, 3.0)
        _set_logit(logits, topk_idx, 40, 2.0)
        _set_logit(logits, topk_idx, 50, 1.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert len(result.token_ids) == 2

        # Greedy request always returns argmax
        greedy_token = result.token_ids[greedy_idx]
        assert greedy_token == 99

        # Top-k request returns one of the top-5 tokens
        topk_token = result.token_ids[topk_idx]
        assert topk_token in {10, 20, 30, 40, 50}

    def test_different_temperatures_in_same_batch(self) -> None:
        """Requests with different temperatures coexist in same microbatch."""
        engine = _make_engine(max_batch_size=4)

        cold_config = SamplingConfiguration(
            do_sample=True, top_k=10, temperature=0.01
        )
        hot_config = SamplingConfiguration(
            do_sample=True, top_k=10, temperature=2.0
        )

        engine.submit_request(
            request_id="cold-req",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=cold_config,
        )
        engine.submit_request(
            request_id="hot-req",
            prompt_tokens=[2],
            max_tokens=10,
            sampling_config=hot_config,
        )

        # Prefill both
        for _ in range(2):
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Create logits with some variation
        logits = _make_logits(2, 50)
        # Give each row a clear max so sampling produces valid results
        _set_logit(logits, 0, 5, 10.0)
        _set_logit(logits, 0, 10, 8.0)
        _set_logit(logits, 0, 15, 6.0)
        _set_logit(logits, 1, 20, 10.0)
        _set_logit(logits, 1, 25, 8.0)
        _set_logit(logits, 1, 30, 6.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert len(result.token_ids) == 2
        # All tokens should be valid vocabulary indices
        for token_id in result.token_ids:
            assert 0 <= token_id < 50


# ---------------------------------------------------------------------------
# Test: Results contain correct slot indices and generations
# ---------------------------------------------------------------------------


class TestSlotMetadata:
    """Verify results contain correct slot indices and generation counters."""

    def test_slot_indices_match_microbatch(self) -> None:
        """Result slot_indices match the active slots in the microbatch."""
        engine = _make_engine(max_batch_size=4)
        config = SamplingConfiguration(do_sample=False)

        # Submit and prefill 2 requests
        for i in range(2):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i + 1],
                max_tokens=10,
                sampling_config=config,
            )

        for i in range(2):
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        active_slots = [s for s in microbatch.slot_states if s.is_active]
        expected_slot_indices = tuple(s.slot_index for s in active_slots)
        expected_slot_generations = tuple(s.slot_generation for s in active_slots)

        logits = _make_logits(2, 50)
        _set_logit(logits, 0, 7, 5.0)
        _set_logit(logits, 1, 13, 5.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert result.slot_indices == expected_slot_indices
        assert result.slot_generations == expected_slot_generations

    def test_slot_generations_are_preserved(self) -> None:
        """Slot generation counters from the microbatch are in the result."""
        engine = _make_engine(max_batch_size=2)
        config = SamplingConfiguration(do_sample=False)

        # Submit and prefill a request
        engine.submit_request(
            request_id="req-1",
            prompt_tokens=[1],
            max_tokens=10,
            sampling_config=config,
        )
        engine.get_prefill_request()
        engine.mark_prefill_done("req-1")

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None
        slot_state = microbatch.slot_states[0]

        # The generation counter should be present in the result
        logits = _make_logits(1, 50)
        _set_logit(logits, 0, 7, 10.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert result.slot_generations == (slot_state.slot_generation,)
        assert result.slot_indices == (slot_state.slot_index,)
        assert result.token_ids == (7,)

    def test_result_tuples_have_consistent_length(self) -> None:
        """All result tuples have the same length as active slot count."""
        engine = _make_engine(max_batch_size=8)
        config = SamplingConfiguration(do_sample=False)

        # Submit 4 requests
        for i in range(4):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[i + 1],
                max_tokens=10,
                sampling_config=config,
            )

        for i in range(4):
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        logits = _make_logits(4, 30)
        for row in range(4):
            _set_logit(logits, row, row * 5, 10.0)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert len(result.token_ids) == 4
        assert len(result.slot_indices) == 4
        assert len(result.slot_generations) == 4


# ---------------------------------------------------------------------------
# Test: Empty microbatch returns empty results
# ---------------------------------------------------------------------------


class TestEmptyMicrobatch:
    """Verify empty microbatch produces empty results."""

    def test_empty_microbatch_returns_empty_result(self) -> None:
        """A microbatch with no active slots returns empty TokenResultBatch."""
        engine = _make_engine()

        # Create an empty microbatch manually
        empty_microbatch = DecodeMicrobatch(
            active_slot_count=0,
            max_batch_size=4,
            slot_states=(),
            input_token_ids=(),
        )

        logits = _make_logits(0, 50)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=empty_microbatch,
            engine=engine,
        )

        assert result.token_ids == ()
        assert result.slot_indices == ()
        assert result.slot_generations == ()

    def test_microbatch_with_inactive_slots_only(self) -> None:
        """A microbatch where all slots are inactive returns empty results."""
        engine = _make_engine()

        # Create microbatch with inactive slots
        inactive_slot = BatchSlotState(
            slot_index=0,
            request_id=None,
            slot_generation=1,
            is_active=False,
        )
        microbatch = DecodeMicrobatch(
            active_slot_count=0,
            max_batch_size=4,
            slot_states=(inactive_slot,),
            input_token_ids=(),
        )

        logits = _make_logits(0, 50)

        result = sample_microbatch(
            logits_batch=logits,
            microbatch=microbatch,
            engine=engine,
        )

        assert result.token_ids == ()
        assert result.slot_indices == ()
        assert result.slot_generations == ()


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify proper error handling for invalid inputs."""

    def test_logits_row_count_mismatch_raises(self) -> None:
        """Mismatched logits rows and active slots raises ValueError."""
        engine = _make_engine()
        config = SamplingConfiguration(do_sample=False)

        _setup_decoding_request(
            engine, "req-1", [1], max_tokens=10, sampling_config=config
        )

        microbatch = engine.get_decode_microbatch()
        assert microbatch is not None

        # Provide wrong number of logit rows (2 rows for 1 active slot)
        wrong_logits = _make_logits(2, 50)

        with pytest.raises(ValueError, match="logits_batch has 2 rows"):
            sample_microbatch(
                logits_batch=wrong_logits,
                microbatch=microbatch,
                engine=engine,
            )
