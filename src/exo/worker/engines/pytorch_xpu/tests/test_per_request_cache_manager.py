"""
Unit tests for PerRequestCacheManager.

Covers:
- Create and retrieve cache by request_id
- Remove cache on completion
- Recycle cache makes states available for reuse
- Multiple concurrent caches are isolated
- Batch position is NOT used as cache key
- GatedDeltaNet state stored by (request_id, layer_index)
- Capacity enforcement
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Load modules with torch mocked out
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ENGINE_DIR = _THIS_DIR.parent
_SAMPLING_PATH = _ENGINE_DIR / "sampling.py"
_BATCHING_PATH = _ENGINE_DIR / "continuous_batching.py"
_CACHE_PATH = _ENGINE_DIR / "gated_deltanet_cache.py"
_STATE_PATH = _ENGINE_DIR / "gated_deltanet_state.py"


def _ensure_torch_mock() -> None:
    """Ensure a minimal torch mock is in sys.modules."""
    if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "Tensor"):
        torch_mock = types.ModuleType("torch")
        torch_mock.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        torch_mock.device = type("device", (), {"__init__": lambda self, *a: None})  # type: ignore[attr-defined]
        torch_mock.float32 = "float32"  # type: ignore[attr-defined]
        torch_mock.bfloat16 = "bfloat16"  # type: ignore[attr-defined]

        # Mock torch.zeros to return a mock tensor with data_ptr and zero_
        def _mock_zeros(*args: object, **kwargs: object) -> MagicMock:
            tensor = MagicMock()
            tensor.data_ptr.return_value = id(tensor)
            tensor.zero_ = MagicMock(return_value=tensor)
            return tensor

        torch_mock.zeros = _mock_zeros  # type: ignore[attr-defined]
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

# Load gated_deltanet_state (dependency of gated_deltanet_cache)
_state_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.gated_deltanet_state", _STATE_PATH
)
GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
GatedDeltaNetStateShape = _state_mod.GatedDeltaNetStateShape

# Load gated_deltanet_cache
_cache_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.gated_deltanet_cache", _CACHE_PATH
)
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache

# Load continuous_batching
_batching_mod = _load_module(
    "exo.worker.engines.pytorch_xpu.continuous_batching", _BATCHING_PATH
)
PerRequestCacheManager = _batching_mod.PerRequestCacheManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_cache() -> MagicMock:
    """Create a mock DynamicCache."""
    cache = MagicMock()
    cache.__len__ = MagicMock(return_value=0)
    return cache


def _make_persistent_state(
    request_id: str, layer_index: int
) -> GatedDeltaNetPersistentState:
    """Create a GatedDeltaNetPersistentState with mock tensors."""
    import torch

    return GatedDeltaNetPersistentState.create(
        request_identifier=request_id,
        layer_index=layer_index,
        batch_size=1,
        num_heads=4,
        key_dim=64,
        value_dim=64,
        conv_dim=512,
        conv_kernel_size=4,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Test: Create and retrieve cache by request_id
# ---------------------------------------------------------------------------


class TestCreateAndRetrieve:
    """Tests for creating and retrieving caches by request_id."""

    def test_create_and_get_cache(self) -> None:
        """A cache can be created and retrieved by request_id."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        dynamic_cache = _make_dynamic_cache()

        manager.create_cache("req-1", dynamic_cache)
        cache = manager.get_cache("req-1")

        assert cache is not None
        assert isinstance(cache, GatedDeltaNetCache)
        assert cache.dynamic_cache is dynamic_cache

    def test_get_nonexistent_returns_none(self) -> None:
        """Getting a cache for an unknown request returns None."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        assert manager.get_cache("nonexistent") is None

    def test_has_cache_true(self) -> None:
        """has_cache returns True for existing caches."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        assert manager.has_cache("req-1") is True

    def test_has_cache_false(self) -> None:
        """has_cache returns False for unknown requests."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        assert manager.has_cache("unknown") is False

    def test_duplicate_create_raises(self) -> None:
        """Creating a cache for an existing request raises ValueError."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        with pytest.raises(ValueError, match="already exists"):
            manager.create_cache("req-1", _make_dynamic_cache())

    def test_capacity_enforcement(self) -> None:
        """Creating beyond max_requests raises RuntimeError."""
        manager = PerRequestCacheManager(max_requests=2, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        manager.create_cache("req-2", _make_dynamic_cache())

        with pytest.raises(RuntimeError, match="at capacity"):
            manager.create_cache("req-3", _make_dynamic_cache())

    def test_active_cache_count(self) -> None:
        """active_cache_count reflects the number of active caches."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        assert manager.active_cache_count == 0

        manager.create_cache("req-1", _make_dynamic_cache())
        assert manager.active_cache_count == 1

        manager.create_cache("req-2", _make_dynamic_cache())
        assert manager.active_cache_count == 2


# ---------------------------------------------------------------------------
# Test: Remove cache on completion
# ---------------------------------------------------------------------------


class TestRemoveCache:
    """Tests for removing caches on request completion."""

    def test_remove_existing_cache(self) -> None:
        """Removing a cache makes it unavailable."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        manager.remove_cache("req-1")
        assert manager.get_cache("req-1") is None
        assert manager.active_cache_count == 0

    def test_remove_nonexistent_is_noop(self) -> None:
        """Removing a nonexistent cache does nothing."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.remove_cache("nonexistent")  # Should not raise
        assert manager.active_cache_count == 0

    def test_remove_clears_gated_deltanet_states(self) -> None:
        """Removing a cache also removes associated GatedDeltaNet states."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        state = _make_persistent_state("req-1", layer_index=3)
        manager.set_gated_deltanet_state("req-1", 3, state)

        assert manager.get_gated_deltanet_state("req-1", 3) is not None

        manager.remove_cache("req-1")
        assert manager.get_gated_deltanet_state("req-1", 3) is None

    def test_remove_frees_capacity(self) -> None:
        """Removing a cache frees capacity for new requests."""
        manager = PerRequestCacheManager(max_requests=2, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        manager.create_cache("req-2", _make_dynamic_cache())

        manager.remove_cache("req-1")
        # Should not raise — capacity freed
        manager.create_cache("req-3", _make_dynamic_cache())
        assert manager.active_cache_count == 2


# ---------------------------------------------------------------------------
# Test: Recycle cache makes states available for reuse
# ---------------------------------------------------------------------------


class TestRecycleCache:
    """Tests for recycling caches and reusing GatedDeltaNet states."""

    def test_recycle_removes_from_active(self) -> None:
        """Recycling a cache removes it from active caches."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        manager.recycle_cache("req-1")
        assert manager.get_cache("req-1") is None
        assert manager.active_cache_count == 0

    def test_recycle_nonexistent_is_noop(self) -> None:
        """Recycling a nonexistent cache does nothing."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.recycle_cache("nonexistent")  # Should not raise

    def test_recycled_states_available_for_claim(self) -> None:
        """Recycled GatedDeltaNet states can be claimed by new requests."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        state = _make_persistent_state("req-1", layer_index=5)
        manager.set_gated_deltanet_state("req-1", 5, state)

        # Recycle req-1's cache
        manager.recycle_cache("req-1")
        assert manager.recycled_state_count == 1

        # Create a new request and claim the recycled state
        manager.create_cache("req-2", _make_dynamic_cache())
        claimed = manager.claim_recycled_state("req-2", layer_index=5)

        assert claimed is not None
        assert claimed is state
        assert claimed.request_identifier == "req-2"
        assert not claimed.is_available_for_reuse
        assert manager.recycled_state_count == 0

    def test_claim_returns_none_when_no_recycled(self) -> None:
        """Claiming when no recycled states exist returns None."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        claimed = manager.claim_recycled_state("req-1", layer_index=5)
        assert claimed is None

    def test_recycle_zeros_state_tensors(self) -> None:
        """Recycled states have their tensors zeroed."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        state = _make_persistent_state("req-1", layer_index=2)
        manager.set_gated_deltanet_state("req-1", 2, state)

        manager.recycle_cache("req-1")

        # The state should be marked as available for reuse
        assert state.is_available_for_reuse
        assert state.request_identifier == ""

    def test_recycle_multiple_layers(self) -> None:
        """Recycling a request with multiple layer states recycles all."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        for layer_idx in [0, 5, 10, 15]:
            state = _make_persistent_state("req-1", layer_index=layer_idx)
            manager.set_gated_deltanet_state("req-1", layer_idx, state)

        manager.recycle_cache("req-1")
        assert manager.recycled_state_count == 4


# ---------------------------------------------------------------------------
# Test: Multiple concurrent caches are isolated
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    """Tests that multiple concurrent caches are fully isolated."""

    def test_separate_caches_per_request(self) -> None:
        """Each request gets its own independent cache."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        dc_1 = _make_dynamic_cache()
        dc_2 = _make_dynamic_cache()

        manager.create_cache("req-1", dc_1)
        manager.create_cache("req-2", dc_2)

        cache_1 = manager.get_cache("req-1")
        cache_2 = manager.get_cache("req-2")

        assert cache_1 is not cache_2
        assert cache_1 is not None
        assert cache_2 is not None
        assert cache_1.dynamic_cache is dc_1
        assert cache_2.dynamic_cache is dc_2

    def test_gated_deltanet_states_isolated(self) -> None:
        """GatedDeltaNet states for different requests are independent."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        manager.create_cache("req-2", _make_dynamic_cache())

        state_1 = _make_persistent_state("req-1", layer_index=3)
        state_2 = _make_persistent_state("req-2", layer_index=3)

        manager.set_gated_deltanet_state("req-1", 3, state_1)
        manager.set_gated_deltanet_state("req-2", 3, state_2)

        retrieved_1 = manager.get_gated_deltanet_state("req-1", 3)
        retrieved_2 = manager.get_gated_deltanet_state("req-2", 3)

        assert retrieved_1 is state_1
        assert retrieved_2 is state_2
        assert retrieved_1 is not retrieved_2

    def test_removing_one_does_not_affect_others(self) -> None:
        """Removing one request's cache leaves others intact."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        manager.create_cache("req-2", _make_dynamic_cache())
        manager.create_cache("req-3", _make_dynamic_cache())

        state_2 = _make_persistent_state("req-2", layer_index=7)
        manager.set_gated_deltanet_state("req-2", 7, state_2)

        manager.remove_cache("req-1")
        manager.remove_cache("req-3")

        # req-2 is unaffected
        assert manager.get_cache("req-2") is not None
        assert manager.get_gated_deltanet_state("req-2", 7) is state_2
        assert manager.active_cache_count == 1

    def test_active_request_ids(self) -> None:
        """active_request_ids lists all requests with active caches."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-a", _make_dynamic_cache())
        manager.create_cache("req-b", _make_dynamic_cache())
        manager.create_cache("req-c", _make_dynamic_cache())

        ids = set(manager.active_request_ids)
        assert ids == {"req-a", "req-b", "req-c"}


# ---------------------------------------------------------------------------
# Test: Batch position is NOT used as cache key
# ---------------------------------------------------------------------------


class TestBatchPositionNotCacheKey:
    """Tests that batch position (slot_index) is not used as cache key.

    The PerRequestCacheManager uses request_id as the stable key.
    Two requests assigned to the same slot at different times must have
    independent caches.
    """

    def test_same_slot_different_requests_independent(self) -> None:
        """Two requests that reuse the same slot have independent caches.

        Simulates: req-1 occupies slot 0, completes, req-2 takes slot 0.
        Their caches must be completely independent.
        """
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)

        # req-1 gets a cache (would be at slot 0 in the scheduler)
        manager.create_cache("req-1", _make_dynamic_cache())
        state_1 = _make_persistent_state("req-1", layer_index=0)
        manager.set_gated_deltanet_state("req-1", 0, state_1)

        # req-1 completes, cache removed
        manager.remove_cache("req-1")

        # req-2 takes the same slot (slot 0) but gets its own cache
        manager.create_cache("req-2", _make_dynamic_cache())
        state_2 = _make_persistent_state("req-2", layer_index=0)
        manager.set_gated_deltanet_state("req-2", 0, state_2)

        # req-2's state is independent — not req-1's state
        retrieved = manager.get_gated_deltanet_state("req-2", 0)
        assert retrieved is state_2
        assert retrieved is not state_1

        # req-1's state is gone
        assert manager.get_gated_deltanet_state("req-1", 0) is None

    def test_no_slot_index_in_api(self) -> None:
        """The cache manager API uses request_id, not slot_index.

        Verifies that all public methods accept request_id (string),
        not numeric batch positions.
        """
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("request-abc-123", _make_dynamic_cache())

        # All lookups use string request_id
        assert manager.get_cache("request-abc-123") is not None
        assert manager.has_cache("request-abc-123") is True

        state = _make_persistent_state("request-abc-123", layer_index=10)
        manager.set_gated_deltanet_state("request-abc-123", 10, state)
        assert manager.get_gated_deltanet_state("request-abc-123", 10) is state

    def test_concurrent_requests_different_slots_isolated(self) -> None:
        """Multiple concurrent requests at different slots are isolated.

        Even though they exist simultaneously, their caches are keyed
        by request_id and do not interfere.
        """
        manager = PerRequestCacheManager(max_requests=8, num_layers=64)

        # Simulate 4 concurrent requests at slots 0-3
        for i in range(4):
            req_id = f"req-{i}"
            manager.create_cache(req_id, _make_dynamic_cache())
            state = _make_persistent_state(req_id, layer_index=0)
            manager.set_gated_deltanet_state(req_id, 0, state)

        # Each request has its own state at layer 0
        states = [
            manager.get_gated_deltanet_state(f"req-{i}", 0)
            for i in range(4)
        ]
        # All states are distinct objects
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert states[i] is not states[j]


# ---------------------------------------------------------------------------
# Test: Memory estimation
# ---------------------------------------------------------------------------


class TestMemoryEstimation:
    """Tests for total_memory_bytes property."""

    def test_empty_manager_zero_bytes(self) -> None:
        """Empty manager reports zero memory."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        assert manager.total_memory_bytes == 0

    def test_memory_increases_with_states(self) -> None:
        """Memory estimate increases as states are added."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        assert manager.total_memory_bytes == 0

        state = _make_persistent_state("req-1", layer_index=0)
        manager.set_gated_deltanet_state("req-1", 0, state)

        # 1 * 4 * 64 * 64 * 4 bytes = 65536 bytes
        expected_bytes = 1 * 4 * 64 * 64 * 4
        assert manager.total_memory_bytes == expected_bytes

    def test_memory_decreases_on_remove(self) -> None:
        """Memory estimate decreases when states are removed."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())

        state = _make_persistent_state("req-1", layer_index=0)
        manager.set_gated_deltanet_state("req-1", 0, state)

        assert manager.total_memory_bytes > 0

        manager.remove_cache("req-1")
        assert manager.total_memory_bytes == 0


# ---------------------------------------------------------------------------
# Test: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation."""

    def test_invalid_max_requests(self) -> None:
        """max_requests must be positive."""
        with pytest.raises(ValueError, match="max_requests must be positive"):
            PerRequestCacheManager(max_requests=0, num_layers=64)

        with pytest.raises(ValueError, match="max_requests must be positive"):
            PerRequestCacheManager(max_requests=-1, num_layers=64)

    def test_invalid_num_layers(self) -> None:
        """num_layers must be positive."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            PerRequestCacheManager(max_requests=4, num_layers=0)

        with pytest.raises(ValueError, match="num_layers must be positive"):
            PerRequestCacheManager(max_requests=4, num_layers=-1)

    def test_set_state_without_cache_raises(self) -> None:
        """Setting state without creating cache first raises ValueError."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        state = _make_persistent_state("req-1", layer_index=0)

        with pytest.raises(ValueError, match="No cache exists"):
            manager.set_gated_deltanet_state("req-1", 0, state)

    def test_set_state_invalid_layer_raises(self) -> None:
        """Setting state with out-of-range layer raises ValueError."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)
        manager.create_cache("req-1", _make_dynamic_cache())
        state = _make_persistent_state("req-1", layer_index=0)

        with pytest.raises(ValueError, match="out of range"):
            manager.set_gated_deltanet_state("req-1", 64, state)

        with pytest.raises(ValueError, match="out of range"):
            manager.set_gated_deltanet_state("req-1", -1, state)

    def test_claim_without_cache_raises(self) -> None:
        """Claiming recycled state without cache raises ValueError."""
        manager = PerRequestCacheManager(max_requests=4, num_layers=64)

        with pytest.raises(ValueError, match="No cache exists"):
            manager.claim_recycled_state("req-1", layer_index=0)
