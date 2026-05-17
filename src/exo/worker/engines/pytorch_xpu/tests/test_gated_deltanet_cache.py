"""
Unit tests for GatedDeltaNetCache integration layer.

Tests that:
- Persistent state is accessible via the cache wrapper
- The keyword ``past_key_values`` (plural) is used consistently
- Full-attention layers still work normally with the cache
- GatedDeltaNet layers can retrieve their persistent state by layer index
- DynamicCache delegation works transparently
- Request lifecycle (reset, recycle, clear) operates correctly

**Validates: Requirements 5.1, 5.2, 5.5, 5.6, 5.7, 5.8**
"""

from __future__ import annotations

import importlib.util
import inspect
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
    "gated_deltanet_state_test_cache",
    _ENGINE_DIR / "gated_deltanet_state.py",
)
_cache_mod = _load_module(
    "gated_deltanet_cache_test",
    _ENGINE_DIR / "gated_deltanet_cache.py",
)

GatedDeltaNetPersistentState = _state_mod.GatedDeltaNetPersistentState
GatedDeltaNetCache = _cache_mod.GatedDeltaNetCache


# ===========================================================================
# Fixtures
# ===========================================================================

# Qwen3.5-4B dimensions for testing
_BATCH_SIZE = 1
_NUM_HEADS = 32
_KEY_DIM = 128
_VALUE_DIM = 128
_CONV_DIM = 256
_CONV_KERNEL_SIZE = 4


def _make_persistent_state(
    request_identifier: str = "test-request-001",
    layer_index: int = 5,
) -> "GatedDeltaNetPersistentState":
    """Create a persistent state container for testing."""
    return GatedDeltaNetPersistentState.create(
        request_identifier=request_identifier,
        layer_index=layer_index,
        batch_size=_BATCH_SIZE,
        num_heads=_NUM_HEADS,
        key_dim=_KEY_DIM,
        value_dim=_VALUE_DIM,
        conv_dim=_CONV_DIM,
        conv_kernel_size=_CONV_KERNEL_SIZE,
        device=torch.device("cpu"),
    )


class FakeDynamicCache:
    """Minimal DynamicCache mock that supports the interface we delegate to.

    This avoids importing transformers (which requires model downloads) while
    testing the delegation behavior.
    """

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._seq_length: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate DynamicCache.update()."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        self._seq_length = key_states.shape[2]
        return key_states, value_states

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.key_cache[index], self.value_cache[index])

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(len(self.key_cache)):
            yield (self.key_cache[i], self.value_cache[i])

    def __repr__(self) -> str:
        return f"FakeDynamicCache(layers={len(self.key_cache)}, seq_len={self._seq_length})"


@pytest.fixture
def fake_dynamic_cache() -> FakeDynamicCache:
    """Create a fake DynamicCache for testing."""
    return FakeDynamicCache()


@pytest.fixture
def cache(fake_dynamic_cache: FakeDynamicCache) -> "GatedDeltaNetCache":
    """Create a GatedDeltaNetCache wrapping a fake DynamicCache."""
    return GatedDeltaNetCache(dynamic_cache=fake_dynamic_cache)


# ===========================================================================
# Tests: Persistent state storage and retrieval
# ===========================================================================


class TestPersistentStateAccess:
    """Test that persistent state is accessible via the cache."""

    def test_get_returns_none_for_unregistered_layer(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """get_gated_deltanet_state returns None for layers without state."""
        assert cache.get_gated_deltanet_state(0) is None
        assert cache.get_gated_deltanet_state(99) is None

    def test_set_and_get_state(self, cache: "GatedDeltaNetCache") -> None:
        """State can be stored and retrieved by layer index."""
        state = _make_persistent_state(layer_index=5)
        cache.set_gated_deltanet_state(5, state)
        retrieved = cache.get_gated_deltanet_state(5)
        assert retrieved is state

    def test_set_multiple_layers(self, cache: "GatedDeltaNetCache") -> None:
        """Multiple layers can have independent persistent states."""
        state_5 = _make_persistent_state(layer_index=5)
        state_10 = _make_persistent_state(layer_index=10)
        state_20 = _make_persistent_state(layer_index=20)

        cache.set_gated_deltanet_state(5, state_5)
        cache.set_gated_deltanet_state(10, state_10)
        cache.set_gated_deltanet_state(20, state_20)

        assert cache.get_gated_deltanet_state(5) is state_5
        assert cache.get_gated_deltanet_state(10) is state_10
        assert cache.get_gated_deltanet_state(20) is state_20

    def test_set_validates_layer_index_mismatch(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """set_gated_deltanet_state raises ValueError on layer_index mismatch."""
        state = _make_persistent_state(layer_index=5)
        with pytest.raises(ValueError, match="does not match"):
            cache.set_gated_deltanet_state(10, state)

    def test_has_gated_deltanet_state(self, cache: "GatedDeltaNetCache") -> None:
        """has_gated_deltanet_state returns correct boolean."""
        assert not cache.has_gated_deltanet_state(5)
        state = _make_persistent_state(layer_index=5)
        cache.set_gated_deltanet_state(5, state)
        assert cache.has_gated_deltanet_state(5)
        assert not cache.has_gated_deltanet_state(6)

    def test_gated_deltanet_layer_indices(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """gated_deltanet_layer_indices returns sorted list."""
        cache.set_gated_deltanet_state(20, _make_persistent_state(layer_index=20))
        cache.set_gated_deltanet_state(5, _make_persistent_state(layer_index=5))
        cache.set_gated_deltanet_state(10, _make_persistent_state(layer_index=10))
        assert cache.gated_deltanet_layer_indices == [5, 10, 20]

    def test_gated_deltanet_state_count(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """gated_deltanet_state_count returns correct count."""
        assert cache.gated_deltanet_state_count == 0
        cache.set_gated_deltanet_state(5, _make_persistent_state(layer_index=5))
        assert cache.gated_deltanet_state_count == 1
        cache.set_gated_deltanet_state(10, _make_persistent_state(layer_index=10))
        assert cache.gated_deltanet_state_count == 2

    def test_remove_gated_deltanet_state(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """remove_gated_deltanet_state removes the state for a layer."""
        state = _make_persistent_state(layer_index=5)
        cache.set_gated_deltanet_state(5, state)
        assert cache.has_gated_deltanet_state(5)
        cache.remove_gated_deltanet_state(5)
        assert not cache.has_gated_deltanet_state(5)

    def test_remove_nonexistent_layer_is_safe(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Removing a non-existent layer does not raise."""
        cache.remove_gated_deltanet_state(99)  # Should not raise


# ===========================================================================
# Tests: past_key_values keyword usage (CRITICAL)
# ===========================================================================


class TestPastKeyValuesKeyword:
    """Verify that the integration uses past_key_values (plural) consistently.

    The singular 'past_key_value' broke the pipeline before — it silently gets
    swallowed by **kwargs and the cache never reaches the layers.
    """

    def test_cache_module_does_not_contain_singular_past_key_value(self) -> None:
        """The cache module source must not contain singular 'past_key_value' as a kwarg name."""
        source = inspect.getsource(_cache_mod)
        # Check that the singular form does not appear as a keyword argument
        # (it's fine in documentation/comments explaining the bug)
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith(("#", '"""', "'''")):
                continue
            if stripped.startswith(('"', "'")):
                continue
            # Check for singular form used as a variable/kwarg assignment
            if "past_key_value" in stripped and "past_key_values" not in stripped:
                # Allow it in string literals (documentation about the bug)
                if '""' in stripped or "''" in stripped:
                    continue
                if "past_key_value``" in stripped or "``past_key_value" in stripped:
                    continue
                # This is a real usage of the singular form — fail
                pytest.fail(
                    f"Line {i} contains singular 'past_key_value' "
                    f"(not in a comment): {stripped!r}"
                )

    def test_pipeline_shard_uses_plural_past_key_values(self) -> None:
        """pipeline_parallel_shard.py must use past_key_values (plural) for cache kwarg."""
        shard_path = _ENGINE_DIR / "pipeline_parallel_shard.py"
        source = shard_path.read_text()
        # The critical line that passes cache to layers
        assert 'layer_kwargs["past_key_values"]' in source
        # Ensure the singular form is NOT used as a kwarg assignment
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Check for singular form used as dict key assignment
            if 'layer_kwargs["past_key_value"]' in stripped:
                pytest.fail(
                    f"Line {i} in pipeline_parallel_shard.py uses singular "
                    f"'past_key_value' as layer kwarg: {stripped!r}"
                )

    def test_cache_wrapper_passable_as_past_key_values(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """The cache wrapper can be passed as past_key_values to a function."""
        # Simulate what pipeline_parallel_shard does
        layer_kwargs: dict[str, object] = {}
        layer_kwargs["past_key_values"] = cache

        # Verify it's accessible
        assert layer_kwargs["past_key_values"] is cache
        # Verify the key is plural
        assert "past_key_values" in layer_kwargs
        assert "past_key_value" not in layer_kwargs


# ===========================================================================
# Tests: DynamicCache delegation (full-attention layers work normally)
# ===========================================================================


class TestDynamicCacheDelegation:
    """Test that full-attention layer operations work through the wrapper."""

    def test_get_seq_length_delegates(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """get_seq_length() is delegated to the underlying DynamicCache."""
        fake_dynamic_cache._seq_length = 42
        assert cache.get_seq_length() == 42

    def test_update_delegates(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """update() is delegated to the underlying DynamicCache."""
        key = torch.randn(1, 8, 5, 64)
        value = torch.randn(1, 8, 5, 64)
        cache.update(key, value, layer_idx=0)
        assert fake_dynamic_cache.key_cache[0] is key
        assert fake_dynamic_cache.value_cache[0] is value

    def test_len_delegates(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """len() is delegated to the underlying DynamicCache."""
        assert len(cache) == 0
        fake_dynamic_cache.key_cache.append(torch.empty(0))
        fake_dynamic_cache.value_cache.append(torch.empty(0))
        assert len(cache) == 1

    def test_getitem_delegates(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """Indexing is delegated to the underlying DynamicCache."""
        key = torch.randn(1, 8, 3, 64)
        value = torch.randn(1, 8, 3, 64)
        fake_dynamic_cache.key_cache.append(key)
        fake_dynamic_cache.value_cache.append(value)
        result = cache[0]
        assert result[0] is key
        assert result[1] is value

    def test_iter_delegates(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """Iteration is delegated to the underlying DynamicCache."""
        key = torch.randn(1, 8, 3, 64)
        value = torch.randn(1, 8, 3, 64)
        fake_dynamic_cache.key_cache.append(key)
        fake_dynamic_cache.value_cache.append(value)
        items = list(cache)
        assert len(items) == 1
        assert items[0][0] is key

    def test_dynamic_cache_property(
        self, cache: "GatedDeltaNetCache", fake_dynamic_cache: FakeDynamicCache
    ) -> None:
        """dynamic_cache property exposes the underlying cache."""
        assert cache.dynamic_cache is fake_dynamic_cache


# ===========================================================================
# Tests: Request lifecycle operations
# ===========================================================================


class TestRequestLifecycle:
    """Test reset, recycle, and clear operations for request management."""

    def test_reset_all_zeros_states(self, cache: "GatedDeltaNetCache") -> None:
        """reset_all_gated_deltanet_states zeros all registered states."""
        state_5 = _make_persistent_state(layer_index=5)
        state_10 = _make_persistent_state(layer_index=10)
        state_5.recurrent_state.fill_(42.0)
        state_10.recurrent_state.fill_(99.0)

        cache.set_gated_deltanet_state(5, state_5)
        cache.set_gated_deltanet_state(10, state_10)

        cache.reset_all_gated_deltanet_states()

        assert torch.all(state_5.recurrent_state == 0)
        assert torch.all(state_10.recurrent_state == 0)
        # States are still registered
        assert cache.gated_deltanet_state_count == 2

    def test_recycle_all_marks_available(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """recycle_all_gated_deltanet_states marks all states as available."""
        state_5 = _make_persistent_state(layer_index=5)
        state_10 = _make_persistent_state(layer_index=10)

        cache.set_gated_deltanet_state(5, state_5)
        cache.set_gated_deltanet_state(10, state_10)

        cache.recycle_all_gated_deltanet_states()

        assert state_5.is_available_for_reuse
        assert state_10.is_available_for_reuse
        # States are still registered (for potential reuse)
        assert cache.gated_deltanet_state_count == 2

    def test_clear_removes_all_states(self, cache: "GatedDeltaNetCache") -> None:
        """clear_gated_deltanet_states removes all state references."""
        cache.set_gated_deltanet_state(5, _make_persistent_state(layer_index=5))
        cache.set_gated_deltanet_state(10, _make_persistent_state(layer_index=10))

        cache.clear_gated_deltanet_states()

        assert cache.gated_deltanet_state_count == 0
        assert cache.get_gated_deltanet_state(5) is None
        assert cache.get_gated_deltanet_state(10) is None


# ===========================================================================
# Tests: Integration scenario — simulating pipeline usage
# ===========================================================================


class TestPipelineIntegration:
    """Test the cache in a scenario mimicking pipeline_parallel_shard usage."""

    def test_full_attention_layer_uses_cache_normally(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Full-attention layers can update the cache via delegation."""
        # Simulate a full-attention layer updating the cache
        key = torch.randn(1, 8, 1, 64)  # (batch, heads, seq_len=1, head_dim)
        value = torch.randn(1, 8, 1, 64)
        cache.update(key, value, layer_idx=0)

        # Verify the update went through
        assert cache.get_seq_length() == 1

    def test_gated_deltanet_layer_retrieves_state(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """GatedDeltaNet layers can retrieve their persistent state from the cache."""
        state = _make_persistent_state(layer_index=3)
        # Simulate writing some state during prefill
        state.recurrent_state.fill_(1.5)
        cache.set_gated_deltanet_state(3, state)

        # Simulate what a GatedDeltaNet layer does during decode
        retrieved = cache.get_gated_deltanet_state(3)
        assert retrieved is not None
        assert retrieved.recurrent_state.dtype == torch.float32
        assert torch.all(retrieved.recurrent_state == 1.5)

    def test_mixed_layer_pipeline(self, cache: "GatedDeltaNetCache") -> None:
        """A pipeline with both full-attention and GatedDeltaNet layers works."""
        # Register GatedDeltaNet states for linear attention layers
        for layer_idx in [1, 2, 4, 5, 7, 8]:
            state = _make_persistent_state(layer_index=layer_idx)
            cache.set_gated_deltanet_state(layer_idx, state)

        # Full-attention layers (0, 3, 6) use the DynamicCache normally
        for layer_idx in [0, 3, 6]:
            key = torch.randn(1, 8, 1, 64)
            value = torch.randn(1, 8, 1, 64)
            cache.update(key, value, layer_idx=layer_idx)

        # Verify both types coexist
        assert cache.gated_deltanet_state_count == 6
        assert cache.get_seq_length() == 1

        # GatedDeltaNet layers can still access their state
        for layer_idx in [1, 2, 4, 5, 7, 8]:
            assert cache.has_gated_deltanet_state(layer_idx)
            state = cache.get_gated_deltanet_state(layer_idx)
            assert state is not None
            assert state.layer_index == layer_idx

    def test_cache_passed_as_past_key_values_kwarg(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """The cache can be passed as past_key_values and accessed by layers."""
        state = _make_persistent_state(layer_index=5)
        cache.set_gated_deltanet_state(5, state)

        # Simulate the layer call pattern from pipeline_parallel_shard.py
        def fake_layer_forward(
            hidden_states: torch.Tensor,
            *,
            past_key_values: object = None,
            **kwargs: object,
        ) -> torch.Tensor:
            """Simulate a decoder layer that accesses past_key_values."""
            assert past_key_values is not None
            # GatedDeltaNet layer would do this:
            assert isinstance(past_key_values, GatedDeltaNetCache)
            retrieved_state = past_key_values.get_gated_deltanet_state(5)
            assert retrieved_state is not None
            assert retrieved_state.recurrent_state.dtype == torch.float32
            return hidden_states

        hidden = torch.randn(1, 1, 2560)
        # CRITICAL: use past_key_values (PLURAL)
        result = fake_layer_forward(hidden, past_key_values=cache)
        assert result is hidden

    def test_request_completion_clears_state(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """On request completion, all GatedDeltaNet state is cleared."""
        for layer_idx in [1, 2, 4, 5]:
            cache.set_gated_deltanet_state(
                layer_idx, _make_persistent_state(layer_index=layer_idx)
            )

        # Request completes
        cache.clear_gated_deltanet_states()

        # All state is gone
        assert cache.gated_deltanet_state_count == 0
        for layer_idx in [1, 2, 4, 5]:
            assert cache.get_gated_deltanet_state(layer_idx) is None


# ===========================================================================
# Tests: repr
# ===========================================================================


class TestRepr:
    """Test string representation."""

    def test_repr_shows_components(
        self, cache: "GatedDeltaNetCache"
    ) -> None:
        """Repr shows both dynamic cache and state count."""
        cache.set_gated_deltanet_state(5, _make_persistent_state(layer_index=5))
        r = repr(cache)
        assert "GatedDeltaNetCache" in r
        assert "1 layers" in r
