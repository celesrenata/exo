"""
Integration test: GatedDeltaNet optimized decode through full shard path.

Exercises the complete pipeline shard lifecycle (initialize_request → prefill →
decode → reset_state) by running the GatedDeltaNet recurrent step through multiple
layers in sequence, comparing optimized output against baseline within tolerance.

This test simulates what PipelineParallelShard does during decode:
1. Creates a GatedDeltaNetCache with persistent state for multiple layers
2. Runs a short prefill (4 tokens) to build up recurrent state
3. Runs multiple decode steps (8 tokens) through all layers sequentially
4. Compares optimized outputs against baseline at every step
5. Verifies the full lifecycle (initialize → decode → reset → reinitialize)

The tolerance is wider than unit tests (rtol=2e-2, atol=2e-2) because bf16
accumulation across multiple layers introduces compounding error.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.7, 5.8, 5.10**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module imports — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_state_mod = _load_module(
    "gated_deltanet_state_shard_integration",
    _ENGINE_DIR / "gated_deltanet_state.py",
)
_cache_mod = _load_module(
    "gated_deltanet_cache_shard_integration",
    _ENGINE_DIR / "gated_deltanet_cache.py",
)
_deltanet_mod = _load_module(
    "gated_deltanet_shard_integration",
    _ENGINE_DIR / "gated_deltanet.py",
)

GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache
initialize_gated_deltanet_state = _state_mod.initialize_gated_deltanet_state
_gated_deltanet_recurrent_step_impl = _deltanet_mod._gated_deltanet_recurrent_step_impl
gated_deltanet_decode_recurrent_step_optimized = (
    _deltanet_mod.gated_deltanet_decode_recurrent_step_optimized
)


# ===========================================================================
# Test constants — Qwen3.5-4B dimensions (smaller for test speed)
# ===========================================================================

_BATCH_SIZE = 1
_NUM_HEADS = 32
_KEY_DIM = 128
_VALUE_DIM = 128
_CONV_DIM = 256
_CONV_KERNEL_SIZE = 4

# Simulate a shard with 6 layers: mix of linear_attention and full_attention
# Qwen3.5 pattern: layers alternate, with linear_attention being the majority
_NUM_LAYERS = 6
_LAYER_TYPES = [
    "linear_attention",  # layer 0
    "linear_attention",  # layer 1
    "full_attention",    # layer 2
    "linear_attention",  # layer 3
    "linear_attention",  # layer 4
    "full_attention",    # layer 5
]

# Integration test tolerance — wider than unit tests due to bf16 accumulation
# across multiple layers compounding error
_RTOL = 2e-2
_ATOL = 2e-2

# Prefill and decode step counts
_PREFILL_TOKENS = 4
_DECODE_STEPS = 8


# ===========================================================================
# Helpers
# ===========================================================================


class FakeDynamicCache:
    """Minimal DynamicCache mock for integration testing."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.key_cache[index], self.value_cache[index])

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(len(self.key_cache)):
            yield (self.key_cache[i], self.value_cache[i])


def _make_random_inputs(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random bf16 inputs for a single decode step.

    Returns (q, k, v, gate, beta) with shapes matching the test constants.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    k = torch.randn(_BATCH_SIZE, _NUM_HEADS, _KEY_DIM, generator=gen, dtype=torch.bfloat16)
    v = torch.randn(_BATCH_SIZE, _NUM_HEADS, _VALUE_DIM, generator=gen, dtype=torch.bfloat16)
    # Gate values should be negative (log-space decay) for realistic behavior
    gate = -torch.abs(
        torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16) * 0.1
    )
    beta = torch.sigmoid(
        torch.randn(_BATCH_SIZE, _NUM_HEADS, generator=gen, dtype=torch.bfloat16)
    )
    return q, k, v, gate, beta


def _init_all_linear_attention_states(
    cache: "GatedDeltaNetCache",
    request_id: str,
) -> list[int]:
    """Initialize persistent state for all linear_attention layers.

    Returns the list of layer indices that have persistent state.
    """
    linear_layer_indices: list[int] = []
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        if layer_type == "linear_attention":
            initialize_gated_deltanet_state(
                cache=cache,
                request_identifier=request_id,
                layer_index=layer_idx,
                batch_size=_BATCH_SIZE,
                num_heads=_NUM_HEADS,
                key_dim=_KEY_DIM,
                value_dim=_VALUE_DIM,
                conv_dim=_CONV_DIM,
                conv_kernel_size=_CONV_KERNEL_SIZE,
                device=torch.device("cpu"),
            )
            linear_layer_indices.append(layer_idx)
    return linear_layer_indices


def _run_prefill_baseline(
    baseline_states: dict[int, torch.Tensor],
    num_tokens: int,
    seed_offset: int = 0,
) -> dict[int, torch.Tensor]:
    """Run prefill through baseline path for all linear_attention layers.

    Processes tokens sequentially through each linear_attention layer,
    accumulating state. Returns the updated baseline states.
    """
    for token_idx in range(num_tokens):
        for layer_idx, layer_type in enumerate(_LAYER_TYPES):
            if layer_type != "linear_attention":
                continue
            seed = seed_offset + token_idx * 1000 + layer_idx * 7
            q, k, v, gate, beta = _make_random_inputs(seed)
            _, baseline_states[layer_idx] = _gated_deltanet_recurrent_step_impl(
                q, k, v, gate, beta, baseline_states[layer_idx]
            )
    return baseline_states


def _run_prefill_optimized(
    cache: "GatedDeltaNetCache",
    num_tokens: int,
    seed_offset: int = 0,
) -> None:
    """Run prefill through optimized path for all linear_attention layers.

    Processes tokens sequentially through each linear_attention layer using
    the optimized persistent state path.
    """
    for token_idx in range(num_tokens):
        for layer_idx, layer_type in enumerate(_LAYER_TYPES):
            if layer_type != "linear_attention":
                continue
            seed = seed_offset + token_idx * 1000 + layer_idx * 7
            q, k, v, gate, beta = _make_random_inputs(seed)
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            gated_deltanet_decode_recurrent_step_optimized(q, k, v, gate, beta, state)


# ===========================================================================
# Integration Tests
# ===========================================================================


@pytest.mark.slow
class TestGatedDeltaNetShardIntegration:
    """Full shard path integration: prefill + decode, optimized vs baseline."""

    def test_prefill_plus_decode_matches_baseline_within_tolerance(self) -> None:
        """Run short prefill + decode through full shard path, compare outputs.

        Simulates the PipelineParallelShard lifecycle:
        1. initialize_request() — creates persistent state for linear_attention layers
        2. Prefill (4 tokens) — builds up recurrent state
        3. Decode (8 tokens) — steady-state generation
        4. Compare outputs at every decode step within tolerance

        The tolerance (rtol=2e-2, atol=2e-2) accounts for bf16 accumulation
        across multiple layers.
        """
        # --- Setup: Optimized path ---
        cache_optimized = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        linear_indices = _init_all_linear_attention_states(
            cache_optimized, "request-shard-001"
        )

        # --- Setup: Baseline path ---
        baseline_states: dict[int, torch.Tensor] = {
            idx: torch.zeros(
                _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
            )
            for idx in linear_indices
        }

        # --- Phase 1: Prefill (4 tokens through all layers) ---
        prefill_seed_offset = 42
        _run_prefill_optimized(cache_optimized, _PREFILL_TOKENS, prefill_seed_offset)
        baseline_states = _run_prefill_baseline(
            baseline_states, _PREFILL_TOKENS, prefill_seed_offset
        )

        # Verify states match after prefill
        for layer_idx in linear_indices:
            opt_state = cache_optimized.get_gated_deltanet_state(layer_idx)
            assert opt_state is not None
            torch.testing.assert_close(
                opt_state.recurrent_state,
                baseline_states[layer_idx],
                rtol=_RTOL,
                atol=_ATOL,
                msg=f"State mismatch after prefill at layer {layer_idx}",
            )

        # --- Phase 2: Decode (8 tokens through all layers) ---
        decode_seed_offset = 10000
        for step in range(_DECODE_STEPS):
            for layer_idx in linear_indices:
                seed = decode_seed_offset + step * 1000 + layer_idx * 13
                q, k, v, gate, beta = _make_random_inputs(seed)

                # Baseline
                output_baseline, baseline_states[layer_idx] = (
                    _gated_deltanet_recurrent_step_impl(
                        q, k, v, gate, beta, baseline_states[layer_idx]
                    )
                )

                # Optimized
                opt_state = cache_optimized.get_gated_deltanet_state(layer_idx)
                assert opt_state is not None
                output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, opt_state
                )

                # Compare outputs within tolerance
                torch.testing.assert_close(
                    output_optimized,
                    output_baseline,
                    rtol=_RTOL,
                    atol=_ATOL,
                    msg=(
                        f"Output mismatch at decode step {step}, layer {layer_idx}"
                    ),
                )

        # --- Final: Verify accumulated states match ---
        for layer_idx in linear_indices:
            opt_state = cache_optimized.get_gated_deltanet_state(layer_idx)
            assert opt_state is not None
            torch.testing.assert_close(
                opt_state.recurrent_state,
                baseline_states[layer_idx],
                rtol=_RTOL,
                atol=_ATOL,
                msg=f"Final state mismatch at layer {layer_idx}",
            )

    def test_initialize_request_and_reset_state_lifecycle(self) -> None:
        """Verify initialize_request() and reset_state() lifecycle methods.

        Exercises the full lifecycle that PipelineParallelShard uses:
        1. Initialize request → persistent states created
        2. Run decode → states accumulate
        3. Reset (recycle) → states zeroed, containers available for reuse
        4. Initialize new request → states claimed by new request
        5. Run decode → fresh state, independent of previous request
        """
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())

        # --- Request 1: Initialize ---
        linear_indices = _init_all_linear_attention_states(cache, "request-A")
        assert cache.gated_deltanet_state_count == len(linear_indices)

        # Verify all states are active for request-A
        for layer_idx in linear_indices:
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            assert state.request_identifier == "request-A"
            assert not state.is_available_for_reuse

        # --- Request 1: Decode (accumulate state) ---
        for step in range(5):
            for layer_idx in linear_indices:
                q, k, v, gate, beta = _make_random_inputs(step * 100 + layer_idx)
                state = cache.get_gated_deltanet_state(layer_idx)
                assert state is not None
                gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, state
                )

        # Verify states are non-zero after decode
        for layer_idx in linear_indices:
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            assert not torch.all(state.recurrent_state == 0)
            assert state.decode_step_count == 5

        # --- Reset (simulates reset_state()) ---
        cache.recycle_all_gated_deltanet_states()

        # Verify states are recycled (zeroed, available for reuse)
        for layer_idx in linear_indices:
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            assert state.is_available_for_reuse
            assert state.request_identifier == ""
            assert torch.all(state.recurrent_state == 0)
            assert state.decode_step_count == 0

        # --- Request 2: Initialize (reuses containers) ---
        data_ptrs_before = {
            idx: cache.get_gated_deltanet_state(idx).recurrent_state.data_ptr()  # type: ignore[union-attr]
            for idx in linear_indices
        }

        # Re-initialize for new request (should claim recycled containers)
        _init_all_linear_attention_states(cache, "request-B")

        # Verify containers were reused (same data_ptr)
        for layer_idx in linear_indices:
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            assert state.request_identifier == "request-B"
            assert not state.is_available_for_reuse
            assert state.recurrent_state.data_ptr() == data_ptrs_before[layer_idx]

        # --- Request 2: Decode (independent of request A) ---
        baseline_states: dict[int, torch.Tensor] = {
            idx: torch.zeros(
                _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
            )
            for idx in linear_indices
        }

        for step in range(3):
            for layer_idx in linear_indices:
                seed = 5000 + step * 100 + layer_idx * 11
                q, k, v, gate, beta = _make_random_inputs(seed)

                # Baseline (fresh state)
                _, baseline_states[layer_idx] = _gated_deltanet_recurrent_step_impl(
                    q, k, v, gate, beta, baseline_states[layer_idx]
                )

                # Optimized
                state = cache.get_gated_deltanet_state(layer_idx)
                assert state is not None
                gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, state
                )

        # Verify request B outputs match baseline (independent of request A)
        for layer_idx in linear_indices:
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            torch.testing.assert_close(
                state.recurrent_state,
                baseline_states[layer_idx],
                rtol=_RTOL,
                atol=_ATOL,
                msg=f"Request B state mismatch at layer {layer_idx}",
            )

    def test_multi_layer_sequential_decode_accumulates_error_within_tolerance(
        self,
    ) -> None:
        """Verify that error does not compound beyond tolerance across layers and steps.

        This test specifically checks that running the optimized path through
        many layers and many decode steps does not accumulate error beyond the
        specified tolerance. The key insight: each layer's output feeds into the
        next layer's input in a real shard, so errors compound.

        We simulate this by using the output of one layer as part of the input
        to the next layer (via a simple linear transform of the output into
        the next layer's query/key/value).
        """
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        linear_indices = _init_all_linear_attention_states(cache, "request-compound")

        baseline_states: dict[int, torch.Tensor] = {
            idx: torch.zeros(
                _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
            )
            for idx in linear_indices
        }

        # Use a fixed initial hidden state that propagates through layers
        gen = torch.Generator().manual_seed(999)
        hidden = torch.randn(
            _BATCH_SIZE, _NUM_HEADS, _VALUE_DIM, generator=gen, dtype=torch.bfloat16
        )

        for step in range(_DECODE_STEPS):
            layer_output_baseline = hidden.clone()
            layer_output_optimized = hidden.clone()

            for layer_idx in linear_indices:
                # Derive q, k, v, gate, beta from the current hidden state
                # This simulates how a real layer transforms hidden_states
                # into attention inputs
                seed = step * 10000 + layer_idx * 37
                rng = torch.Generator().manual_seed(seed)

                # Project hidden into q, k (key_dim) and v (value_dim)
                # Use deterministic projections so baseline and optimized get same inputs
                q_proj = torch.randn(
                    _VALUE_DIM, _KEY_DIM, generator=rng, dtype=torch.bfloat16
                )
                k_proj = torch.randn(
                    _VALUE_DIM, _KEY_DIM, generator=rng, dtype=torch.bfloat16
                )

                q = torch.matmul(layer_output_baseline, q_proj)  # (B, H, key_dim)
                k = torch.matmul(layer_output_baseline, k_proj)  # (B, H, key_dim)
                v = layer_output_baseline  # (B, H, value_dim)
                gate = -torch.abs(
                    torch.randn(
                        _BATCH_SIZE, _NUM_HEADS, generator=rng, dtype=torch.bfloat16
                    )
                    * 0.05
                )
                beta = torch.sigmoid(
                    torch.randn(
                        _BATCH_SIZE, _NUM_HEADS, generator=rng, dtype=torch.bfloat16
                    )
                )

                # Baseline path
                output_baseline, baseline_states[layer_idx] = (
                    _gated_deltanet_recurrent_step_impl(
                        q, k, v, gate, beta, baseline_states[layer_idx]
                    )
                )

                # Optimized path (same inputs)
                q_opt = torch.matmul(layer_output_optimized, q_proj)
                k_opt = torch.matmul(layer_output_optimized, k_proj)
                v_opt = layer_output_optimized

                opt_state = cache.get_gated_deltanet_state(layer_idx)
                assert opt_state is not None
                output_optimized = gated_deltanet_decode_recurrent_step_optimized(
                    q_opt, k_opt, v_opt, gate, beta, opt_state
                )

                # Propagate output to next layer
                layer_output_baseline = output_baseline
                layer_output_optimized = output_optimized

            # After all layers, compare final outputs for this step
            torch.testing.assert_close(
                layer_output_optimized,
                layer_output_baseline,
                rtol=_RTOL,
                atol=_ATOL,
                msg=f"Compounded output mismatch at decode step {step}",
            )

            # Update hidden for next step (use baseline output as ground truth)
            hidden = layer_output_baseline

    def test_optimized_flag_disabled_produces_independent_baseline(self) -> None:
        """When optimization is disabled, the shard uses baseline behavior.

        This test verifies that the flag-based dispatch works correctly:
        - With flag enabled: uses persistent state (optimized path)
        - With flag disabled: would use standard path (no persistent state)

        We verify this by checking that the cache has no persistent states
        when the flag would be disabled (simulated by not calling
        initialize_gated_deltanet_state).
        """
        # Simulate "disabled" path: cache exists but no persistent states
        cache_disabled = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        assert cache_disabled.gated_deltanet_state_count == 0

        # Simulate "enabled" path: cache with persistent states
        cache_enabled = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        linear_indices = _init_all_linear_attention_states(
            cache_enabled, "request-flag-test"
        )
        assert cache_enabled.gated_deltanet_state_count == len(linear_indices)

        # Run baseline (no persistent state — uses _gated_deltanet_recurrent_step_impl)
        baseline_state = torch.zeros(
            _BATCH_SIZE, _NUM_HEADS, _KEY_DIM, _VALUE_DIM, dtype=torch.float32
        )
        baseline_outputs: list[torch.Tensor] = []

        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(step * 77)
            output, baseline_state = _gated_deltanet_recurrent_step_impl(
                q, k, v, gate, beta, baseline_state
            )
            baseline_outputs.append(output)

        # Run optimized (with persistent state)
        optimized_outputs: list[torch.Tensor] = []
        # Use first linear_attention layer
        first_linear_idx = linear_indices[0]

        for step in range(5):
            q, k, v, gate, beta = _make_random_inputs(step * 77)
            state = cache_enabled.get_gated_deltanet_state(first_linear_idx)
            assert state is not None
            output = gated_deltanet_decode_recurrent_step_optimized(
                q, k, v, gate, beta, state
            )
            optimized_outputs.append(output)

        # Both paths produce equivalent results
        for step_idx, (opt_out, base_out) in enumerate(
            zip(optimized_outputs, baseline_outputs, strict=True)
        ):
            torch.testing.assert_close(
                opt_out,
                base_out,
                rtol=_RTOL,
                atol=_ATOL,
                msg=f"Flag test: output mismatch at step {step_idx}",
            )

    def test_reset_then_decode_produces_fresh_results(self) -> None:
        """After reset, decode produces results identical to a fresh start.

        This verifies that reset_state() properly clears all accumulated state
        so that a subsequent decode session is not contaminated by the previous one.
        """
        # --- First session: accumulate state ---
        cache = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        linear_indices = _init_all_linear_attention_states(cache, "request-first")

        for step in range(10):
            for layer_idx in linear_indices:
                q, k, v, gate, beta = _make_random_inputs(step * 50 + layer_idx)
                state = cache.get_gated_deltanet_state(layer_idx)
                assert state is not None
                gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, state
                )

        # --- Reset (simulates reset_state()) ---
        cache.recycle_all_gated_deltanet_states()
        _init_all_linear_attention_states(cache, "request-second")

        # --- Second session: decode from fresh state ---
        optimized_outputs: list[torch.Tensor] = []
        for step in range(5):
            for layer_idx in linear_indices:
                seed = 9000 + step * 100 + layer_idx * 3
                q, k, v, gate, beta = _make_random_inputs(seed)
                state = cache.get_gated_deltanet_state(layer_idx)
                assert state is not None
                output = gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, state
                )
                optimized_outputs.append(output)

        # --- Fresh baseline (never had first session) ---
        cache_fresh = GatedDeltaNetCache(dynamic_cache=FakeDynamicCache())
        _init_all_linear_attention_states(cache_fresh, "request-fresh")

        fresh_outputs: list[torch.Tensor] = []
        for step in range(5):
            for layer_idx in linear_indices:
                seed = 9000 + step * 100 + layer_idx * 3
                q, k, v, gate, beta = _make_random_inputs(seed)
                state = cache_fresh.get_gated_deltanet_state(layer_idx)
                assert state is not None
                output = gated_deltanet_decode_recurrent_step_optimized(
                    q, k, v, gate, beta, state
                )
                fresh_outputs.append(output)

        # Outputs after reset must exactly match fresh start
        for idx, (reset_out, fresh_out) in enumerate(
            zip(optimized_outputs, fresh_outputs, strict=True)
        ):
            torch.testing.assert_close(
                reset_out,
                fresh_out,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Reset contamination detected at output index {idx}",
            )
