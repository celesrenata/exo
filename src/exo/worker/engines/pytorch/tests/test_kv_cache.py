"""Unit tests for model/kv_cache.py — KV cache management.

Tests verify allocation, update, get, clear, and dynamic growth
using CPU tensors (no GPU hardware required for unit tests).
"""

from __future__ import annotations

import torch
import pytest

from exo.worker.engines.pytorch.model.kv_cache import KVCache


class TestKVCacheInit:
    """Tests for KVCache initialization."""

    def test_default_dtype_is_float16(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            max_seq_len=128,
            device="cpu",
        )
        assert cache.dtype == torch.float16

    def test_custom_dtype(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            max_seq_len=128,
            device="cpu",
            dtype=torch.float32,
        )
        assert cache.dtype == torch.float32

    def test_initial_seq_len_is_zero(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=64,
            max_seq_len=128,
            device="cpu",
        )
        assert cache.current_seq_len == 0


class TestKVCacheAllocate:
    """Tests for KVCache.allocate()."""

    def test_allocates_correct_number_of_layers(self) -> None:
        cache = KVCache(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=256,
            device="cpu",
        )
        cache.allocate()

        # Should be able to get from all 4 layers
        for layer_idx in range(4):
            key, value = cache.get(layer_idx, seq_len=0)
            assert key.shape == (8, 0, 64)
            assert value.shape == (8, 0, 64)

    def test_allocates_on_specified_device(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=4,
            head_dim=32,
            max_seq_len=64,
            device="cpu",
        )
        cache.allocate()

        key, value = cache.get(0, seq_len=1)
        assert key.device.type == "cpu"
        assert value.device.type == "cpu"

    def test_resets_seq_len_on_reallocate(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=16,
            max_seq_len=32,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # Write some data
        key = torch.randn(2, 1, 16)
        value = torch.randn(2, 1, 16)
        cache.update(0, key, value, position=0)
        assert cache.current_seq_len == 1

        # Re-allocate should reset
        cache.allocate()
        assert cache.current_seq_len == 0


class TestKVCacheUpdate:
    """Tests for KVCache.update()."""

    def test_single_token_update(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        key = torch.ones(2, 1, 4)
        value = torch.ones(2, 1, 4) * 2.0
        cache.update(0, key, value, position=0)

        assert cache.current_seq_len == 1

        cached_key, cached_value = cache.get(0, seq_len=1)
        assert torch.allclose(cached_key, key)
        assert torch.allclose(cached_value, value)

    def test_multi_token_update(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # Write 5 tokens at once
        key = torch.randn(2, 5, 4)
        value = torch.randn(2, 5, 4)
        cache.update(0, key, value, position=0)

        assert cache.current_seq_len == 5

        cached_key, cached_value = cache.get(0, seq_len=5)
        assert torch.allclose(cached_key, key)
        assert torch.allclose(cached_value, value)

    def test_sequential_updates(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # First token
        key1 = torch.ones(2, 1, 4) * 1.0
        value1 = torch.ones(2, 1, 4) * 10.0
        cache.update(0, key1, value1, position=0)

        # Second token
        key2 = torch.ones(2, 1, 4) * 2.0
        value2 = torch.ones(2, 1, 4) * 20.0
        cache.update(0, key2, value2, position=1)

        assert cache.current_seq_len == 2

        cached_key, cached_value = cache.get(0, seq_len=2)
        assert torch.allclose(cached_key[:, 0:1, :], key1)
        assert torch.allclose(cached_key[:, 1:2, :], key2)
        assert torch.allclose(cached_value[:, 0:1, :], value1)
        assert torch.allclose(cached_value[:, 1:2, :], value2)

    def test_raises_without_allocation(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
        )

        key = torch.ones(2, 1, 4)
        value = torch.ones(2, 1, 4)

        with pytest.raises(RuntimeError, match="not been allocated"):
            cache.update(0, key, value, position=0)

    def test_raises_on_invalid_layer_index(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
        )
        cache.allocate()

        key = torch.ones(2, 1, 4)
        value = torch.ones(2, 1, 4)

        with pytest.raises(IndexError, match="out of range"):
            cache.update(5, key, value, position=0)

        with pytest.raises(IndexError, match="out of range"):
            cache.update(-1, key, value, position=0)


class TestKVCacheGet:
    """Tests for KVCache.get()."""

    def test_returns_correct_slice(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        key = torch.randn(2, 5, 4)
        value = torch.randn(2, 5, 4)
        cache.update(0, key, value, position=0)

        # Get only first 3 positions
        cached_key, cached_value = cache.get(0, seq_len=3)
        assert cached_key.shape == (2, 3, 4)
        assert cached_value.shape == (2, 3, 4)
        assert torch.allclose(cached_key, key[:, :3, :])
        assert torch.allclose(cached_value, value[:, :3, :])

    def test_get_with_none_seq_len_uses_current(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        key = torch.randn(2, 3, 4)
        value = torch.randn(2, 3, 4)
        cache.update(0, key, value, position=0)

        cached_key, cached_value = cache.get(0)
        assert cached_key.shape == (2, 3, 4)
        assert cached_value.shape == (2, 3, 4)

    def test_raises_without_allocation(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="not been allocated"):
            cache.get(0, seq_len=1)

    def test_raises_on_invalid_layer_index(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
        )
        cache.allocate()

        with pytest.raises(IndexError, match="out of range"):
            cache.get(3, seq_len=1)


class TestKVCacheDynamicGrowth:
    """Tests for dynamic cache growth when position exceeds allocation."""

    def test_grows_when_position_exceeds_max(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=4,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # Write at position 0 (within bounds)
        key1 = torch.ones(2, 1, 4) * 1.0
        value1 = torch.ones(2, 1, 4) * 10.0
        cache.update(0, key1, value1, position=0)

        # Write at position 5 (exceeds max_seq_len=4)
        key2 = torch.ones(2, 1, 4) * 2.0
        value2 = torch.ones(2, 1, 4) * 20.0
        cache.update(0, key2, value2, position=5)

        assert cache.current_seq_len == 6
        assert cache.max_seq_len >= 6

        # Verify both writes are preserved
        cached_key, cached_value = cache.get(0, seq_len=6)
        assert torch.allclose(cached_key[:, 0:1, :], key1)
        assert torch.allclose(cached_key[:, 5:6, :], key2)
        assert torch.allclose(cached_value[:, 0:1, :], value1)
        assert torch.allclose(cached_value[:, 5:6, :], value2)

    def test_growth_preserves_existing_data(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=4,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # Fill both layers with data
        key0 = torch.randn(2, 3, 4)
        value0 = torch.randn(2, 3, 4)
        cache.update(0, key0, value0, position=0)

        key1 = torch.randn(2, 3, 4)
        value1 = torch.randn(2, 3, 4)
        cache.update(1, key1, value1, position=0)

        # Trigger growth on layer 0
        key_new = torch.randn(2, 1, 4)
        value_new = torch.randn(2, 1, 4)
        cache.update(0, key_new, value_new, position=10)

        # Verify original data in both layers is preserved
        cached_key0, cached_value0 = cache.get(0, seq_len=3)
        assert torch.allclose(cached_key0, key0)
        assert torch.allclose(cached_value0, value0)

        cached_key1, cached_value1 = cache.get(1, seq_len=3)
        assert torch.allclose(cached_key1, key1)
        assert torch.allclose(cached_value1, value1)

    def test_growth_at_least_doubles(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=8,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        # Write at position 9 (just past max_seq_len=8)
        key = torch.ones(2, 1, 4)
        value = torch.ones(2, 1, 4)
        cache.update(0, key, value, position=9)

        # Should have grown to at least 16 (double of 8)
        assert cache.max_seq_len >= 16


class TestKVCacheClear:
    """Tests for KVCache.clear()."""

    def test_resets_seq_len(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        key = torch.randn(2, 5, 4)
        value = torch.randn(2, 5, 4)
        cache.update(0, key, value, position=0)
        assert cache.current_seq_len == 5

        cache.clear()
        assert cache.current_seq_len == 0

    def test_zeros_cache_data(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()

        key = torch.ones(2, 3, 4) * 99.0
        value = torch.ones(2, 3, 4) * 99.0
        cache.update(0, key, value, position=0)

        cache.clear()

        # After clear, getting data should return zeros
        cached_key, cached_value = cache.get(0, seq_len=3)
        assert torch.allclose(cached_key, torch.zeros(2, 3, 4))
        assert torch.allclose(cached_value, torch.zeros(2, 3, 4))

    def test_clear_without_allocation_is_noop(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
        )
        # Should not raise
        cache.clear()

    def test_preserves_allocation(self) -> None:
        cache = KVCache(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_seq_len=16,
            device="cpu",
            dtype=torch.float32,
        )
        cache.allocate()
        original_max = cache.max_seq_len

        cache.clear()

        # Should still be able to write without re-allocating
        key = torch.randn(2, 1, 4)
        value = torch.randn(2, 1, 4)
        cache.update(0, key, value, position=0)
        assert cache.max_seq_len == original_max
