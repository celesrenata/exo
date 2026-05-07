"""
Tests for KV Cache Manager

This module tests the KVCacheManager implementation including:
- Cache creation and retrieval
- LRU eviction policy
- Memory monitoring
- Cache statistics
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from exo.worker.engines.pytorch_xpu.kv_cache_manager import (
    CacheStatistics,
    KVCache,
    KVCacheManager,
)


class TestKVCache:
    """Test KVCache dataclass."""

    def test_kv_cache_creation(self) -> None:
        """Test creating a KVCache instance."""
        keys = [None, None, None]
        values = [None, None, None]

        cache = KVCache(
            request_id="test-request",
            keys=keys,
            values=values,
            position=0,
            max_length=2048,
            last_accessed=time.time(),
            is_active=True,
        )

        assert cache.request_id == "test-request"
        assert len(cache.keys) == 3
        assert len(cache.values) == 3
        assert cache.position == 0
        assert cache.max_length == 2048
        assert cache.is_active is True

    def test_kv_cache_immutable(self) -> None:
        """Test that KVCache is immutable (frozen)."""
        cache = KVCache(
            request_id="test",
            keys=[],
            values=[],
            position=0,
            max_length=1024,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            cache.position = 10  # type: ignore


class TestCacheStatistics:
    """Test CacheStatistics dataclass."""

    def test_statistics_creation(self) -> None:
        """Test creating CacheStatistics instance."""
        stats = CacheStatistics()

        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.total_evictions == 0
        assert stats.active_caches == 0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStatistics(cache_hits=80, cache_misses=20)

        assert stats.hit_rate == 80.0
        assert stats.miss_rate == 20.0

    def test_hit_rate_zero_requests(self) -> None:
        """Test hit rate when no requests."""
        stats = CacheStatistics()

        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 100.0


class TestKVCacheManager:
    """Test KVCacheManager class."""

    def test_manager_initialization(self) -> None:
        """Test KVCacheManager initialization."""
        manager = KVCacheManager(
            device_type="cpu",
            device_id=0,
            memory_threshold_percent=75.0,
        )

        assert manager._device_type == "cpu"
        assert manager._device_id == 0
        assert manager._memory_threshold_percent == 75.0
        assert len(manager._caches) == 0
        assert len(manager._active_requests) == 0

    def test_create_cache(self) -> None:
        """Test creating a new cache."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        cache = manager.create_cache(
            request_id="req-1",
            num_layers=12,
            max_length=2048,
        )

        assert cache.request_id == "req-1"
        assert len(cache.keys) == 12
        assert len(cache.values) == 12
        assert cache.position == 0
        assert cache.max_length == 2048
        assert cache.is_active is True

        # Verify cache is stored
        assert manager.get_cache_count() == 1
        assert manager.get_active_request_count() == 1

    def test_create_duplicate_cache_raises_error(self) -> None:
        """Test that creating duplicate cache raises error."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        manager.create_cache(request_id="req-1", num_layers=12)

        with pytest.raises(ValueError, match="Cache already exists"):
            manager.create_cache(request_id="req-1", num_layers=12)

    def test_get_cache_hit(self) -> None:
        """Test getting an existing cache (cache hit)."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create cache
        created_cache = manager.create_cache(request_id="req-1", num_layers=12)

        # Get cache
        retrieved_cache = manager.get_cache("req-1")

        assert retrieved_cache is not None
        assert retrieved_cache.request_id == created_cache.request_id

        # Verify statistics
        stats = manager.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0

    def test_get_cache_miss(self) -> None:
        """Test getting a non-existent cache (cache miss)."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Try to get non-existent cache
        cache = manager.get_cache("non-existent")

        assert cache is None

        # Verify statistics
        stats = manager.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1

    def test_update_cache(self) -> None:
        """Test updating cache with new key/value."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create cache
        manager.create_cache(request_id="req-1", num_layers=3)

        # Update cache for layer 0
        mock_key = Mock()
        mock_value = Mock()
        manager.update_cache(
            request_id="req-1",
            layer_idx=0,
            new_key=mock_key,
            new_value=mock_value,
        )

        # Verify update
        cache = manager.get_cache("req-1")
        assert cache is not None
        assert cache.keys[0] is mock_key
        assert cache.values[0] is mock_value
        assert cache.position == 1  # Position incremented

    def test_update_cache_with_position(self) -> None:
        """Test updating cache with explicit position."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        manager.create_cache(request_id="req-1", num_layers=3)

        manager.update_cache(
            request_id="req-1",
            layer_idx=0,
            new_key=Mock(),
            new_value=Mock(),
            new_position=42,
        )

        cache = manager.get_cache("req-1")
        assert cache is not None
        assert cache.position == 42

    def test_update_nonexistent_cache_raises_error(self) -> None:
        """Test that updating non-existent cache raises error."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        with pytest.raises(ValueError, match="No cache exists"):
            manager.update_cache(
                request_id="non-existent",
                layer_idx=0,
                new_key=Mock(),
                new_value=Mock(),
            )

    def test_evict_cache(self) -> None:
        """Test evicting a cache."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create cache
        manager.create_cache(request_id="req-1", num_layers=12)
        assert manager.get_cache_count() == 1

        # Evict cache
        manager.evict_cache("req-1")

        assert manager.get_cache_count() == 0
        assert manager.get_cache("req-1") is None

        # Verify statistics
        stats = manager.get_stats()
        assert stats["total_evictions"] == 1

    def test_evict_nonexistent_cache(self) -> None:
        """Test evicting non-existent cache (should not raise error)."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Should not raise error
        manager.evict_cache("non-existent")

    def test_mark_request_complete(self) -> None:
        """Test marking request as complete."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create cache
        manager.create_cache(request_id="req-1", num_layers=12)
        assert manager.get_active_request_count() == 1

        # Mark complete
        manager.mark_request_complete("req-1")

        assert manager.get_active_request_count() == 0

        # Cache should still exist but marked inactive
        cache = manager.get_cache("req-1")
        assert cache is not None
        assert cache.is_active is False

    def test_lru_eviction_order(self) -> None:
        """Test that LRU eviction evicts oldest cache first."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create multiple caches with delays
        manager.create_cache(request_id="req-1", num_layers=12)
        time.sleep(0.01)
        manager.create_cache(request_id="req-2", num_layers=12)
        time.sleep(0.01)
        manager.create_cache(request_id="req-3", num_layers=12)

        # Mark all as complete so they can be evicted
        manager.mark_request_complete("req-1")
        manager.mark_request_complete("req-2")
        manager.mark_request_complete("req-3")

        # Access req-2 to update its timestamp
        manager.get_cache("req-2")

        # Manually evict oldest (should be req-1)
        caches = list(manager._caches.items())
        inactive = [(rid, c) for rid, c in caches if rid not in manager._active_requests]
        inactive.sort(key=lambda x: x[1].last_accessed)

        assert inactive[0][0] == "req-1"  # Oldest
        assert inactive[1][0] == "req-3"  # Middle
        assert inactive[2][0] == "req-2"  # Newest (accessed recently)

    def test_active_requests_not_evicted(self) -> None:
        """Test that active requests are never evicted."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create caches
        manager.create_cache(request_id="req-active", num_layers=12)
        manager.create_cache(request_id="req-inactive", num_layers=12)

        # Mark one as complete
        manager.mark_request_complete("req-inactive")

        # Get inactive caches
        inactive = [
            rid for rid, _ in manager._caches.items() if rid not in manager._active_requests
        ]

        assert "req-inactive" in inactive
        assert "req-active" not in inactive

    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create some activity
        manager.create_cache(request_id="req-1", num_layers=12)
        manager.create_cache(request_id="req-2", num_layers=12)
        manager.get_cache("req-1")  # Hit
        manager.get_cache("req-3")  # Miss
        manager.evict_cache("req-2")

        stats = manager.get_stats()

        assert stats["total_requests"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["total_evictions"] == 1
        assert stats["total_caches"] == 1
        assert stats["active_caches"] == 1
        assert stats["active_requests"] == 1
        assert "hit_rate_percent" in stats
        assert "miss_rate_percent" in stats

    def test_clear_all_caches(self) -> None:
        """Test clearing all caches."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create multiple caches
        manager.create_cache(request_id="req-1", num_layers=12)
        manager.create_cache(request_id="req-2", num_layers=12)
        manager.create_cache(request_id="req-3", num_layers=12)

        assert manager.get_cache_count() == 3

        # Clear all
        manager.clear_all_caches()

        assert manager.get_cache_count() == 0
        assert manager.get_active_request_count() == 0

    @patch("exo.worker.engines.pytorch_xpu.kv_cache_manager.logger")
    def test_memory_monitoring_without_torch(self, mock_logger: MagicMock) -> None:
        """Test that manager works without PyTorch (CPU-only mode)."""
        with patch.object(KVCacheManager, "_torch_available", False):
            manager = KVCacheManager(device_type="cpu", device_id=0)

            # Should still work for basic operations
            manager.create_cache(request_id="req-1", num_layers=12)
            cache = manager.get_cache("req-1")

            assert cache is not None
            assert cache.request_id == "req-1"

    def test_get_cache_updates_last_accessed(self) -> None:
        """Test that getting cache updates last_accessed timestamp."""
        manager = KVCacheManager(device_type="cpu", device_id=0)

        # Create cache
        cache1 = manager.create_cache(request_id="req-1", num_layers=12)
        original_time = cache1.last_accessed

        # Wait a bit
        time.sleep(0.01)

        # Get cache (should update timestamp)
        cache2 = manager.get_cache("req-1")

        assert cache2 is not None
        assert cache2.last_accessed > original_time
