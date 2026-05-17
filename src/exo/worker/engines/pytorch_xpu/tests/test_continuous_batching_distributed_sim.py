"""
Simulated distributed integration tests for continuous batching.

These tests exercise the full continuous batching pipeline using the
ContinuousBatchingEngine and ContinuousDecodeStep driver without actual
distributed communication. They simulate the multi-request concurrent
behavior that the distributed system would exhibit across 4 ranks.

Tests verify:
1. Submit 8 requests with varying max_tokens, all complete correctly
2. Per-request token order is monotonically increasing (no out-of-order)
3. Aggregate throughput improves over single-request sequential execution

A true distributed integration test requires 4 ranks with Gloo communication
and can only run on the gremlin cluster. These tests validate the scheduling
and batching logic that drives the distributed pipeline.

Marked @pytest.mark.slow since they run multi-step sessions.
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
run_continuous_decode_steps = _decode_mod.run_continuous_decode_steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLING = SamplingConfiguration(do_sample=False)

# Simulated 4-rank pipeline: each decode step has a fixed cost representing
# the time for activations to traverse all 4 pipeline stages.
_SIMULATED_RANKS = 4
_NUM_LAYERS = 16  # 4 layers per rank in simulation


def _make_engine(max_batch_size: int = 8) -> ContinuousBatchingEngine:
    """Create an engine configured for the simulated distributed test."""
    return ContinuousBatchingEngine(
        max_batch_size=max_batch_size,
        num_layers=_NUM_LAYERS,
        world_size=_SIMULATED_RANKS,
    )


# ---------------------------------------------------------------------------
# Test 1: Submit 8 requests, verify all complete
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEightRequestsAllComplete:
    """Submit 8 requests with varying max_tokens (2-8), verify all complete."""

    def test_all_eight_requests_complete(self) -> None:
        """All 8 requests complete with the correct number of tokens."""
        engine = _make_engine(max_batch_size=8)
        step = ContinuousDecodeStep(engine)

        # Submit 8 requests with varying max_tokens from 2 to 8
        request_configs: list[tuple[str, int]] = [
            ("req-0", 2),
            ("req-1", 3),
            ("req-2", 4),
            ("req-3", 5),
            ("req-4", 6),
            ("req-5", 7),
            ("req-6", 8),
            ("req-7", 4),
        ]

        # Track tokens generated per request via callbacks
        tokens_per_request: dict[str, list[int]] = {
            f"req-{i}": [] for i in range(8)
        }

        def on_token_callback(request_id: str, token_id: int) -> None:
            tokens_per_request[request_id].append(token_id)

        for request_id, max_tokens in request_configs:
            engine.submit_request(
                request_id=request_id,
                prompt_tokens=[100 + i for i in range(5)],
                max_tokens=max_tokens,
                sampling_config=_DEFAULT_SAMPLING,
                on_token=on_token_callback,
            )

        # Token counter for generating unique token IDs
        token_counter = 1000

        def execute_decode(microbatch: object) -> None:
            nonlocal token_counter
            mb = microbatch  # type: ignore[assignment]
            token_ids_list: list[int] = []
            slot_indices_list: list[int] = []
            slot_generations_list: list[int] = []

            for slot_state in mb.slot_states:
                token_counter += 1
                token_ids_list.append(token_counter)
                slot_indices_list.append(slot_state.slot_index)
                slot_generations_list.append(slot_state.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids_list),
                slot_indices=tuple(slot_indices_list),
                slot_generations=tuple(slot_generations_list),
            )
            step.report_decode_results(results)

        result = run_continuous_decode_steps(
            step=step,
            execute_decode=execute_decode,
        )

        # Verify all 8 requests completed
        assert len(result.completed_requests) == 8
        assert set(result.completed_requests) == {
            f"req-{i}" for i in range(8)
        }

        # Verify each request received the correct number of tokens
        for request_id, max_tokens in request_configs:
            assert len(tokens_per_request[request_id]) == max_tokens, (
                f"{request_id} expected {max_tokens} tokens, "
                f"got {len(tokens_per_request[request_id])}"
            )

        # Verify total tokens generated matches sum of all max_tokens
        expected_total = sum(mt for _, mt in request_configs)
        assert result.total_tokens_generated == expected_total

    def test_no_cancelled_requests(self) -> None:
        """No requests are cancelled during normal execution."""
        engine = _make_engine(max_batch_size=8)
        step = ContinuousDecodeStep(engine)

        for i in range(8):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[1, 2, 3],
                max_tokens=3,
                sampling_config=_DEFAULT_SAMPLING,
            )

        token_counter = 0

        def execute_decode(microbatch: object) -> None:
            nonlocal token_counter
            mb = microbatch  # type: ignore[assignment]
            token_ids_list: list[int] = []
            slot_indices_list: list[int] = []
            slot_generations_list: list[int] = []

            for slot_state in mb.slot_states:
                token_counter += 1
                token_ids_list.append(token_counter)
                slot_indices_list.append(slot_state.slot_index)
                slot_generations_list.append(slot_state.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids_list),
                slot_indices=tuple(slot_indices_list),
                slot_generations=tuple(slot_generations_list),
            )
            step.report_decode_results(results)

        result = run_continuous_decode_steps(
            step=step,
            execute_decode=execute_decode,
        )

        assert result.cancelled_requests == []
        assert len(result.completed_requests) == 8


# ---------------------------------------------------------------------------
# Test 2: Verify per-request token order
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPerRequestTokenOrder:
    """Each request receives tokens in monotonically increasing step order."""

    def test_tokens_arrive_in_generation_order(self) -> None:
        """Token IDs per request are in the order they were generated.

        Each decode step assigns a monotonically increasing step counter.
        Tokens for each request must arrive in step order — no out-of-order
        delivery.
        """
        engine = _make_engine(max_batch_size=8)
        step = ContinuousDecodeStep(engine)

        # Submit 8 requests with varying lengths
        max_tokens_list = [3, 5, 4, 6, 2, 7, 8, 3]
        for i, max_tokens in enumerate(max_tokens_list):
            engine.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[10, 20, 30],
                max_tokens=max_tokens,
                sampling_config=_DEFAULT_SAMPLING,
            )

        # Track (step_number, token_id) per request
        token_order: dict[str, list[tuple[int, int]]] = {
            f"req-{i}": [] for i in range(8)
        }
        global_step = 0

        def execute_decode(microbatch: object) -> None:
            nonlocal global_step
            global_step += 1
            mb = microbatch  # type: ignore[assignment]
            token_ids_list: list[int] = []
            slot_indices_list: list[int] = []
            slot_generations_list: list[int] = []

            for slot_state in mb.slot_states:
                # Token ID encodes the global step for traceability
                token_id = global_step * 100 + slot_state.slot_index
                token_ids_list.append(token_id)
                slot_indices_list.append(slot_state.slot_index)
                slot_generations_list.append(slot_state.slot_generation)

                # Record the order for this request
                if slot_state.request_id is not None:
                    token_order[slot_state.request_id].append(
                        (global_step, token_id)
                    )

            results = TokenResultBatch(
                token_ids=tuple(token_ids_list),
                slot_indices=tuple(slot_indices_list),
                slot_generations=tuple(slot_generations_list),
            )
            step.report_decode_results(results)

        run_continuous_decode_steps(
            step=step,
            execute_decode=execute_decode,
        )

        # Verify per-request token order: step numbers are strictly increasing
        for request_id, entries in token_order.items():
            if not entries:
                continue
            step_numbers = [entry[0] for entry in entries]
            for j in range(1, len(step_numbers)):
                assert step_numbers[j] > step_numbers[j - 1], (
                    f"{request_id}: out-of-order delivery at position {j}, "
                    f"step {step_numbers[j]} <= {step_numbers[j - 1]}"
                )

    def test_no_duplicate_tokens_per_request(self) -> None:
        """No request receives the same token_id twice in a session."""
        engine2 = _make_engine(max_batch_size=8)
        step2 = ContinuousDecodeStep(engine2)
        tokens_received2: dict[str, list[int]] = {
            f"req-{i}": [] for i in range(8)
        }

        def on_token_callback2(request_id: str, token_id: int) -> None:
            tokens_received2[request_id].append(token_id)

        for i in range(8):
            engine2.submit_request(
                request_id=f"req-{i}",
                prompt_tokens=[1, 2],
                max_tokens=5,
                sampling_config=_DEFAULT_SAMPLING,
                on_token=on_token_callback2,
            )

        unique_counter = 0

        def execute_decode2(microbatch: object) -> None:
            nonlocal unique_counter
            mb = microbatch  # type: ignore[assignment]
            token_ids_list: list[int] = []
            slot_indices_list: list[int] = []
            slot_generations_list: list[int] = []

            for slot_state in mb.slot_states:
                unique_counter += 1
                token_ids_list.append(unique_counter)
                slot_indices_list.append(slot_state.slot_index)
                slot_generations_list.append(slot_state.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids_list),
                slot_indices=tuple(slot_indices_list),
                slot_generations=tuple(slot_generations_list),
            )
            step2.report_decode_results(results)

        run_continuous_decode_steps(
            step=step2,
            execute_decode=execute_decode2,
        )

        # Verify no duplicates per request
        for request_id, tokens in tokens_received2.items():
            assert len(tokens) == len(set(tokens)), (
                f"{request_id}: received duplicate tokens: {tokens}"
            )


# ---------------------------------------------------------------------------
# Test 3: Verify aggregate throughput improves over single request
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAggregateThroughputImprovement:
    """Batched execution processes more total tokens per step than sequential."""

    def test_batched_throughput_exceeds_sequential(self) -> None:
        """Compare sequential vs batched: batched has higher tokens-per-step.

        Sequential: run 8 requests one at a time. Each decode step processes
        exactly 1 token (batch size = 1).

        Batched: run 8 requests concurrently. Each decode step processes
        up to 8 tokens (batch size = number of active requests).

        The batched mode achieves higher aggregate throughput because it
        processes multiple requests per decode step.
        """
        max_tokens_per_request = 5
        num_requests = 8

        # --- Sequential execution: one request at a time ---
        sequential_total_steps = 0

        def _run_single_request(request_id: str, max_tok: int) -> int:
            """Run a single request and return the number of decode steps."""
            eng = _make_engine(max_batch_size=8)
            stp = ContinuousDecodeStep(eng)

            eng.submit_request(
                request_id=request_id,
                prompt_tokens=[1, 2, 3],
                max_tokens=max_tok,
                sampling_config=_DEFAULT_SAMPLING,
            )

            def _decode(microbatch: object) -> None:
                mb = microbatch  # type: ignore[assignment]
                token_ids_list: list[int] = []
                slot_indices_list: list[int] = []
                slot_generations_list: list[int] = []

                for slot_state in mb.slot_states:
                    token_ids_list.append(42)
                    slot_indices_list.append(slot_state.slot_index)
                    slot_generations_list.append(slot_state.slot_generation)

                results = TokenResultBatch(
                    token_ids=tuple(token_ids_list),
                    slot_indices=tuple(slot_indices_list),
                    slot_generations=tuple(slot_generations_list),
                )
                stp.report_decode_results(results)

            res = run_continuous_decode_steps(
                step=stp,
                execute_decode=_decode,
            )
            return res.total_steps

        for i in range(num_requests):
            sequential_total_steps += _run_single_request(
                f"seq-req-{i}", max_tokens_per_request
            )

        sequential_total_tokens = num_requests * max_tokens_per_request
        sequential_tokens_per_step = (
            sequential_total_tokens / sequential_total_steps
        )

        # --- Batched execution: all 8 requests concurrently ---
        engine_batch = _make_engine(max_batch_size=8)
        step_batch = ContinuousDecodeStep(engine_batch)

        for i in range(num_requests):
            engine_batch.submit_request(
                request_id=f"batch-req-{i}",
                prompt_tokens=[1, 2, 3],
                max_tokens=max_tokens_per_request,
                sampling_config=_DEFAULT_SAMPLING,
            )

        def execute_decode_batch(microbatch: object) -> None:
            mb = microbatch  # type: ignore[assignment]
            token_ids_list: list[int] = []
            slot_indices_list: list[int] = []
            slot_generations_list: list[int] = []

            for slot_state in mb.slot_states:
                token_ids_list.append(42)
                slot_indices_list.append(slot_state.slot_index)
                slot_generations_list.append(slot_state.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids_list),
                slot_indices=tuple(slot_indices_list),
                slot_generations=tuple(slot_generations_list),
            )
            step_batch.report_decode_results(results)

        result_batch = run_continuous_decode_steps(
            step=step_batch,
            execute_decode=execute_decode_batch,
        )

        batched_total_tokens = result_batch.total_tokens_generated
        batched_total_steps = result_batch.total_steps
        batched_tokens_per_step = batched_total_tokens / batched_total_steps

        # Verify both produced the same total tokens
        assert batched_total_tokens == sequential_total_tokens

        # Verify batched throughput is higher
        assert batched_tokens_per_step > sequential_tokens_per_step, (
            f"Batched tokens/step ({batched_tokens_per_step:.2f}) should "
            f"exceed sequential tokens/step ({sequential_tokens_per_step:.2f})"
        )

        # Sequential should be exactly 1.0 tokens/step (one request at a time)
        assert sequential_tokens_per_step == 1.0

        # Batched should be significantly higher (close to num_requests
        # when all requests have the same max_tokens)
        assert batched_tokens_per_step > 1.0

    def test_batched_fewer_total_steps(self) -> None:
        """Batched execution uses fewer total decode steps than sequential.

        With 8 requests of 5 tokens each:
        - Sequential: 8 × 5 = 40 decode steps total
        - Batched: 5 decode steps (all 8 requests batched together)

        The batched mode is more efficient because it amortizes the per-step
        overhead across multiple requests.
        """
        max_tokens_per_request = 5
        num_requests = 8

        # Sequential: each request runs independently
        sequential_total_steps = 0

        def _run_single(request_id: str, max_tok: int) -> int:
            """Run a single request and return the number of decode steps."""
            eng = _make_engine(max_batch_size=8)
            stp = ContinuousDecodeStep(eng)

            eng.submit_request(
                request_id=request_id,
                prompt_tokens=[1],
                max_tokens=max_tok,
                sampling_config=_DEFAULT_SAMPLING,
            )

            def _decode(microbatch: object) -> None:
                mb = microbatch  # type: ignore[assignment]
                token_ids = tuple(42 for _ in mb.slot_states)
                slot_indices = tuple(
                    s.slot_index for s in mb.slot_states
                )
                slot_generations = tuple(
                    s.slot_generation for s in mb.slot_states
                )
                results = TokenResultBatch(
                    token_ids=token_ids,
                    slot_indices=slot_indices,
                    slot_generations=slot_generations,
                )
                stp.report_decode_results(results)

            res = run_continuous_decode_steps(
                step=stp,
                execute_decode=_decode,
            )
            return res.total_steps

        for i in range(num_requests):
            sequential_total_steps += _run_single(
                f"seq-{i}", max_tokens_per_request
            )

        # Batched: all requests run concurrently
        engine_batch = _make_engine(max_batch_size=8)
        step_batch = ContinuousDecodeStep(engine_batch)

        for i in range(num_requests):
            engine_batch.submit_request(
                request_id=f"batch-{i}",
                prompt_tokens=[1],
                max_tokens=max_tokens_per_request,
                sampling_config=_DEFAULT_SAMPLING,
            )

        def execute_decode_batch(microbatch: object) -> None:
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
            step_batch.report_decode_results(results)

        result_batch = run_continuous_decode_steps(
            step=step_batch,
            execute_decode=execute_decode_batch,
        )

        # Sequential: 8 requests × 5 tokens = 40 steps
        assert sequential_total_steps == num_requests * max_tokens_per_request

        # Batched: 5 steps (all 8 complete in parallel)
        assert result_batch.total_steps == max_tokens_per_request

        # Batched uses strictly fewer steps
        assert result_batch.total_steps < sequential_total_steps

    def test_varying_lengths_still_improves_throughput(self) -> None:
        """Even with varying max_tokens, batched throughput is higher.

        When requests have different lengths, shorter requests complete
        first and the batch size decreases over time. The average
        tokens-per-step is still higher than sequential (1.0).
        """
        # Varying max_tokens: 2, 3, 4, 5, 6, 7, 8, 4
        max_tokens_list = [2, 3, 4, 5, 6, 7, 8, 4]
        total_tokens = sum(max_tokens_list)

        # Sequential baseline
        sequential_total_steps = sum(max_tokens_list)  # one token per step

        # Batched execution
        engine_batch = _make_engine(max_batch_size=8)
        step_batch = ContinuousDecodeStep(engine_batch)

        for i, max_tokens in enumerate(max_tokens_list):
            engine_batch.submit_request(
                request_id=f"var-req-{i}",
                prompt_tokens=[1, 2],
                max_tokens=max_tokens,
                sampling_config=_DEFAULT_SAMPLING,
            )

        def execute_decode_batch(microbatch: object) -> None:
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
            step_batch.report_decode_results(results)

        result_batch = run_continuous_decode_steps(
            step=step_batch,
            execute_decode=execute_decode_batch,
        )

        # Verify all tokens generated
        assert result_batch.total_tokens_generated == total_tokens

        # Batched tokens per step
        batched_tokens_per_step = (
            result_batch.total_tokens_generated / result_batch.total_steps
        )

        # Sequential tokens per step is always 1.0
        sequential_tokens_per_step = total_tokens / sequential_total_steps

        # Batched is better
        assert batched_tokens_per_step > sequential_tokens_per_step
        assert batched_tokens_per_step > 1.0

        # Batched uses fewer steps than sequential
        assert result_batch.total_steps < sequential_total_steps

        # The maximum number of steps in batched mode equals the longest
        # request (8 tokens), since all requests start together
        assert result_batch.total_steps == max(max_tokens_list)
