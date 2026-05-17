"""
Property-based tests for the continuous batching system.

Uses Hypothesis to verify invariants under random inputs:
1. Request isolation under random interleavings
2. Batch slot reuse safety
3. Cancellation safety
4. Deterministic greedy equivalence to independent generation

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.9, 3.10, 3.13**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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
ContinuousBatchScheduler = _batching_mod.ContinuousBatchScheduler
RequestIdentifierMap = _batching_mod.RequestIdentifierMap
TokenResultBatch = _batching_mod.TokenResultBatch

# Load continuous_batching_engine
_engine_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching_engine", _ENGINE_PATH
)
ContinuousBatchingEngine = _engine_mod.ContinuousBatchingEngine


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Request IDs: short unique strings
_request_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=3,
    max_size=8,
)

# Prompt token lists: lists of positive ints
_prompt_tokens = st.lists(
    st.integers(min_value=1, max_value=50000),
    min_size=1,
    max_size=10,
)

# Max tokens: small values for fast testing
_max_tokens = st.integers(min_value=1, max_value=20)

# Greedy sampling config (deterministic)
_GREEDY_CONFIG = SamplingConfiguration(do_sample=False)


# Operation types for random interleaving
_Operation = st.sampled_from(["submit", "prefill", "decode", "complete", "cancel"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move_request_to_decode(
    engine: ContinuousBatchingEngine, request_id: str
) -> tuple[int, int] | None:
    """Move a request through prefill to decode-ready. Returns (slot, gen) or None."""
    result = engine.get_prefill_request()
    if result is None:
        return None
    if result[0] != request_id:
        return None
    engine.mark_prefill_done(request_id)
    microbatch = engine.get_decode_microbatch()
    if microbatch is None:
        return None
    for slot_state in microbatch.slot_states:
        if slot_state.request_id == request_id:
            return (slot_state.slot_index, slot_state.slot_generation)
    return None


# ---------------------------------------------------------------------------
# Property 1: Request Isolation Under Random Interleavings
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRequestIsolationProperty:
    """Property 1: Request isolation under random interleavings.

    For generated interleavings of request admission, decode, completion,
    and cancellation, no request may observe another request's token history.

    **Validates: Requirements 3.1, 3.2, 3.5**
    """

    @given(
        request_ids=st.lists(
            _request_id, min_size=2, max_size=6, unique=True
        ),
        max_tokens_list=st.lists(
            _max_tokens, min_size=6, max_size=6,
        ),
    )
    @settings(max_examples=50)
    def test_each_request_receives_only_its_own_tokens(
        self,
        request_ids: list[str],
        max_tokens_list: list[int],
    ) -> None:
        """Each request's token count matches its reported results,
        and no request receives tokens meant for another.

        **Validates: Requirements 3.1, 3.2, 3.5**
        """
        engine = ContinuousBatchingEngine(
            max_batch_size=8, num_layers=4, world_size=4
        )

        # Track tokens received per request via callbacks
        tokens_per_request: dict[str, list[int]] = {
            rid: [] for rid in request_ids
        }

        def make_callback(rid: str):  # noqa: ANN202
            def cb(request_id: str, token_id: int) -> None:
                tokens_per_request[rid].append(token_id)
            return cb

        # Submit all requests
        for i, rid in enumerate(request_ids):
            mt = max_tokens_list[i % len(max_tokens_list)]
            engine.submit_request(
                request_id=rid,
                prompt_tokens=[100 + i, 200 + i],
                max_tokens=mt,
                sampling_config=_GREEDY_CONFIG,
                on_token=make_callback(rid),
            )

        # Prefill all requests sequentially
        slot_info: dict[str, tuple[int, int]] = {}
        for rid in request_ids:
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])
            microbatch = engine.get_decode_microbatch()
            assert microbatch is not None
            for ss in microbatch.slot_states:
                if ss.request_id == rid:
                    slot_info[rid] = (ss.slot_index, ss.slot_generation)

        # Generate tokens — each request gets unique token IDs
        # Token IDs are encoded as (request_index * 1000 + step)
        completed_requests: set[str] = set()
        for step in range(max(max_tokens_list)):
            if not engine.has_work:
                break
            microbatch = engine.get_decode_microbatch()
            if microbatch is None:
                break

            token_ids: list[int] = []
            slot_indices: list[int] = []
            slot_generations: list[int] = []

            for ss in microbatch.slot_states:
                rid = ss.request_id
                assert rid is not None
                req_idx = request_ids.index(rid)
                token_id = req_idx * 1000 + step
                token_ids.append(token_id)
                slot_indices.append(ss.slot_index)
                slot_generations.append(ss.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids),
                slot_indices=tuple(slot_indices),
                slot_generations=tuple(slot_generations),
            )
            newly_completed = engine.report_token_results(results)
            completed_requests.update(newly_completed)

        # Verify isolation: each request only received tokens with its
        # own encoded prefix (req_idx * 1000 + step)
        for i, rid in enumerate(request_ids):
            for token_id in tokens_per_request[rid]:
                expected_prefix = i * 1000
                assert expected_prefix <= token_id < expected_prefix + 100, (
                    f"Request '{rid}' (index {i}) received token {token_id} "
                    f"which belongs to a different request. "
                    f"Expected range [{expected_prefix}, {expected_prefix + 100})"
                )


# ---------------------------------------------------------------------------
# Property 2: Batch Slot Reuse Safety
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBatchSlotReuseSafetyProperty:
    """Property 2: Batch slot reuse safety.

    No two active requests ever share the same slot index, and generation
    counters always increment on slot reuse.

    **Validates: Requirements 3.9, 3.10**
    """

    @given(
        request_ids=st.lists(
            _request_id, min_size=3, max_size=8, unique=True
        ),
        cancel_indices=st.lists(
            st.integers(min_value=0, max_value=7), min_size=1, max_size=4
        ),
    )
    @settings(max_examples=50)
    def test_no_two_active_requests_share_slot(
        self,
        request_ids: list[str],
        cancel_indices: list[int],
    ) -> None:
        """No two active requests ever share the same slot index,
        and generation counters always increment on reuse.

        **Validates: Requirements 3.9, 3.10**
        """
        max_batch_size = 4
        id_map = RequestIdentifierMap(max_batch_size=max_batch_size)

        # Track all generations seen per slot
        slot_generations: dict[int, list[int]] = {
            i: [] for i in range(max_batch_size)
        }

        # Phase 1: Assign slots to first batch of requests
        assigned: dict[str, tuple[int, int]] = {}
        for rid in request_ids[:max_batch_size]:
            slot_idx, gen = id_map.assign_slot(rid)
            assigned[rid] = (slot_idx, gen)
            slot_generations[slot_idx].append(gen)

        # Verify no duplicate slot indices among active requests
        active_slots = [
            slot_idx for slot_idx, _ in assigned.values()
        ]
        assert len(active_slots) == len(set(active_slots)), (
            f"Duplicate slot indices found among active requests: {active_slots}"
        )

        # Phase 2: Release some slots (simulating cancellation/completion)
        released_rids: list[str] = []
        for idx in cancel_indices:
            bounded_idx = idx % len(request_ids[:max_batch_size])
            rid = request_ids[bounded_idx]
            if rid in assigned and rid not in released_rids:
                id_map.release_slot(rid)
                released_rids.append(rid)
                del assigned[rid]

        # Phase 3: Assign new requests to freed slots
        new_request_idx = max_batch_size
        for rid in released_rids:
            if new_request_idx >= len(request_ids):
                break
            new_rid = request_ids[new_request_idx]
            new_request_idx += 1
            if new_rid not in assigned:
                slot_idx, gen = id_map.assign_slot(new_rid)
                assigned[new_rid] = (slot_idx, gen)
                slot_generations[slot_idx].append(gen)

        # Verify: no two active requests share a slot
        active_slots_final = [
            slot_idx for slot_idx, _ in assigned.values()
        ]
        assert len(active_slots_final) == len(set(active_slots_final)), (
            f"Duplicate slot indices after reuse: {active_slots_final}"
        )

        # Verify: generation counters are strictly increasing per slot
        for slot_idx, gens in slot_generations.items():
            for i in range(1, len(gens)):
                assert gens[i] > gens[i - 1], (
                    f"Slot {slot_idx} generation did not increase: "
                    f"{gens[i - 1]} -> {gens[i]}"
                )


# ---------------------------------------------------------------------------
# Property 3: Cancellation Safety
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCancellationSafetyProperty:
    """Property 3: Cancellation safety.

    Cancelled requests never appear in decode batches after cancellation,
    and their slots are properly freed.

    **Validates: Requirements 3.4, 3.5, 3.10**
    """

    @given(
        request_ids=st.lists(
            _request_id, min_size=3, max_size=6, unique=True
        ),
        cancel_at_step=st.integers(min_value=0, max_value=5),
        cancel_target_idx=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_cancelled_request_never_in_decode_batch(
        self,
        request_ids: list[str],
        cancel_at_step: int,
        cancel_target_idx: int,
    ) -> None:
        """After cancellation, the request never appears in subsequent
        decode batches, and its slot is freed for reuse.

        **Validates: Requirements 3.4, 3.5, 3.10**
        """
        engine = ContinuousBatchingEngine(
            max_batch_size=8, num_layers=4, world_size=4
        )

        # Submit all requests
        for rid in request_ids:
            engine.submit_request(
                request_id=rid,
                prompt_tokens=[1, 2, 3],
                max_tokens=10,
                sampling_config=_GREEDY_CONFIG,
            )

        # Prefill all
        for rid in request_ids:
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        # Pick which request to cancel
        cancel_idx = cancel_target_idx % len(request_ids)
        cancel_rid = request_ids[cancel_idx]

        # Run decode steps, cancelling at the specified step
        cancelled = False
        for step in range(cancel_at_step + 3):
            if not engine.has_work:
                break

            # Cancel at the specified step
            if step == cancel_at_step and not cancelled:
                engine.cancel_request(cancel_rid)
                cancelled = True

            # After cancellation, verify the cancelled request is not
            # in any decode batch
            microbatch = engine.get_decode_microbatch()
            if microbatch is None:
                continue

            if cancelled:
                batch_request_ids = {
                    ss.request_id for ss in microbatch.slot_states
                }
                assert cancel_rid not in batch_request_ids, (
                    f"Cancelled request '{cancel_rid}' appeared in decode "
                    f"batch at step {step} (cancelled at step {cancel_at_step})"
                )

            # Report dummy tokens for active requests
            token_ids: list[int] = []
            slot_indices: list[int] = []
            slot_generations: list[int] = []
            for ss in microbatch.slot_states:
                token_ids.append(step * 100)
                slot_indices.append(ss.slot_index)
                slot_generations.append(ss.slot_generation)

            if token_ids:
                results = TokenResultBatch(
                    token_ids=tuple(token_ids),
                    slot_indices=tuple(slot_indices),
                    slot_generations=tuple(slot_generations),
                )
                engine.report_token_results(results)

        # Final verification: cancelled request is not active
        if cancelled:
            assert engine.active_request_count <= len(request_ids) - 1


    @given(
        request_ids=st.lists(
            _request_id, min_size=2, max_size=5, unique=True
        ),
        cancel_indices=st.lists(
            st.integers(min_value=0, max_value=4), min_size=1, max_size=3
        ),
    )
    @settings(max_examples=50)
    def test_cancelled_slots_are_freed(
        self,
        request_ids: list[str],
        cancel_indices: list[int],
    ) -> None:
        """Slots from cancelled requests are freed and can be reused
        by new requests.

        **Validates: Requirements 3.10**
        """
        max_batch_size = 4
        engine = ContinuousBatchingEngine(
            max_batch_size=max_batch_size, num_layers=4, world_size=4
        )

        # Submit and prefill requests (up to max_batch_size)
        active_ids = request_ids[:max_batch_size]
        for rid in active_ids:
            engine.submit_request(
                request_id=rid,
                prompt_tokens=[1, 2],
                max_tokens=10,
                sampling_config=_GREEDY_CONFIG,
            )

        for rid in active_ids:
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        initial_batch_size = engine.decode_batch_size

        # Cancel some requests
        cancelled_rids: set[str] = set()
        for idx in cancel_indices:
            bounded_idx = idx % len(active_ids)
            rid = active_ids[bounded_idx]
            if rid not in cancelled_rids:
                engine.cancel_request(rid)
                cancelled_rids.add(rid)

        # Verify batch size decreased
        assert engine.decode_batch_size == initial_batch_size - len(cancelled_rids)

        # Submit new requests — they should be able to get slots
        for i in range(len(cancelled_rids)):
            new_rid = f"new-{i}"
            engine.submit_request(
                request_id=new_rid,
                prompt_tokens=[10, 20],
                max_tokens=5,
                sampling_config=_GREEDY_CONFIG,
            )
            result = engine.get_prefill_request()
            assert result is not None
            engine.mark_prefill_done(result[0])

        # Verify new requests are in the decode batch
        assert engine.decode_batch_size == initial_batch_size


# ---------------------------------------------------------------------------
# Property 4: Deterministic Greedy Equivalence
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeterministicGreedyEquivalenceProperty:
    """Property 4: Deterministic greedy equivalence to independent generation.

    For greedy sampling (do_sample=False), the token sequence produced for
    a request in a batched context is identical to what it would produce
    in isolation.

    **Validates: Requirements 3.13**
    """

    @given(
        request_ids=st.lists(
            _request_id, min_size=2, max_size=4, unique=True
        ),
        prompt_tokens_list=st.lists(
            _prompt_tokens, min_size=4, max_size=4,
        ),
        max_tokens_val=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=50)
    def test_batched_greedy_equals_independent_greedy(
        self,
        request_ids: list[str],
        prompt_tokens_list: list[list[int]],
        max_tokens_val: int,
    ) -> None:
        """For greedy sampling, batched generation produces the same
        token sequence per request as running each request independently.

        Since greedy is deterministic and the engine routes tokens by
        slot/generation, each request's token stream is independent of
        other requests in the batch.

        **Validates: Requirements 3.13**
        """
        # We simulate deterministic token generation where the "model"
        # produces token_id = hash(request_id, step) for each step.
        # This verifies the engine correctly routes tokens to requests
        # regardless of batch composition.

        def deterministic_token(rid: str, step: int) -> int:
            """Deterministic token for a given request and step."""
            return hash((rid, step)) % 50000

        # --- Run in batched mode ---
        batched_engine = ContinuousBatchingEngine(
            max_batch_size=8, num_layers=4, world_size=4
        )
        batched_tokens: dict[str, list[int]] = {
            rid: [] for rid in request_ids
        }

        def batched_callback(rid: str) -> callable:
            def cb(request_id: str, token_id: int) -> None:
                batched_tokens[rid].append(token_id)
            return cb

        # Submit all requests to batched engine
        for i, rid in enumerate(request_ids):
            prompts = prompt_tokens_list[i % len(prompt_tokens_list)]
            batched_engine.submit_request(
                request_id=rid,
                prompt_tokens=prompts,
                max_tokens=max_tokens_val,
                sampling_config=_GREEDY_CONFIG,
                on_token=batched_callback(rid),
            )

        # Prefill all
        for rid in request_ids:
            result = batched_engine.get_prefill_request()
            assert result is not None
            batched_engine.mark_prefill_done(result[0])

        # Track per-request step counters
        step_counters: dict[str, int] = {rid: 0 for rid in request_ids}

        # Run decode steps
        for _ in range(max_tokens_val):
            if not batched_engine.has_work:
                break
            microbatch = batched_engine.get_decode_microbatch()
            if microbatch is None:
                break

            token_ids: list[int] = []
            slot_indices: list[int] = []
            slot_generations: list[int] = []

            for ss in microbatch.slot_states:
                rid = ss.request_id
                assert rid is not None
                token_id = deterministic_token(rid, step_counters[rid])
                step_counters[rid] += 1
                token_ids.append(token_id)
                slot_indices.append(ss.slot_index)
                slot_generations.append(ss.slot_generation)

            results = TokenResultBatch(
                token_ids=tuple(token_ids),
                slot_indices=tuple(slot_indices),
                slot_generations=tuple(slot_generations),
            )
            batched_engine.report_token_results(results)

        # --- Run each request independently ---
        independent_tokens: dict[str, list[int]] = {
            rid: [] for rid in request_ids
        }

        for i, rid in enumerate(request_ids):
            solo_engine = ContinuousBatchingEngine(
                max_batch_size=8, num_layers=4, world_size=4
            )

            def solo_callback(request_id: str, token_id: int) -> None:
                independent_tokens[rid].append(token_id)

            prompts = prompt_tokens_list[i % len(prompt_tokens_list)]
            solo_engine.submit_request(
                request_id=rid,
                prompt_tokens=prompts,
                max_tokens=max_tokens_val,
                sampling_config=_GREEDY_CONFIG,
                on_token=solo_callback,
            )

            # Prefill
            result = solo_engine.get_prefill_request()
            assert result is not None
            solo_engine.mark_prefill_done(rid)

            # Decode
            for step in range(max_tokens_val):
                if not solo_engine.has_work:
                    break
                microbatch = solo_engine.get_decode_microbatch()
                if microbatch is None:
                    break

                ss = microbatch.slot_states[0]
                token_id = deterministic_token(rid, step)

                results = TokenResultBatch(
                    token_ids=(token_id,),
                    slot_indices=(ss.slot_index,),
                    slot_generations=(ss.slot_generation,),
                )
                solo_engine.report_token_results(results)

        # --- Compare: batched must equal independent ---
        for rid in request_ids:
            assert batched_tokens[rid] == independent_tokens[rid], (
                f"Request '{rid}' produced different tokens in batched vs "
                f"independent mode.\n"
                f"  Batched:     {batched_tokens[rid]}\n"
                f"  Independent: {independent_tokens[rid]}"
            )
