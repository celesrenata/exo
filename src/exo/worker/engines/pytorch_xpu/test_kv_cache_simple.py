#!/usr/bin/env python3
"""
Simple test script for KV Cache Manager

This script validates the basic functionality of the KVCacheManager
without requiring pytest or other test frameworks.
"""

import time

from kv_cache_manager import KVCacheManager


def test_basic_operations() -> None:
    """Test basic cache operations."""
    print("Testing basic cache operations...")

    manager = KVCacheManager(device_type="cpu", device_id=0)

    # Test 1: Create cache
    print("  Creating cache...")
    cache = manager.create_cache(request_id="req-1", num_layers=12, max_length=2048)
    assert cache.request_id == "req-1"
    assert len(cache.keys) == 12
    assert len(cache.values) == 12
    print("  ✓ Cache created successfully")

    # Test 2: Get cache (hit)
    print("  Testing cache hit...")
    retrieved = manager.get_cache("req-1")
    assert retrieved is not None
    assert retrieved.request_id == "req-1"
    print("  ✓ Cache hit successful")

    # Test 3: Get cache (miss)
    print("  Testing cache miss...")
    missing = manager.get_cache("non-existent")
    assert missing is None
    print("  ✓ Cache miss handled correctly")

    # Test 4: Update cache
    print("  Testing cache update...")
    manager.update_cache(
        request_id="req-1",
        layer_idx=0,
        new_key="test_key",
        new_value="test_value",
    )
    updated = manager.get_cache("req-1")
    assert updated is not None
    assert updated.keys[0] == "test_key"
    assert updated.values[0] == "test_value"
    assert updated.position == 1
    print("  ✓ Cache updated successfully")

    # Test 5: Statistics
    print("  Testing statistics...")
    stats = manager.get_stats()
    assert stats["total_requests"] == 1
    assert stats["cache_hits"] == 2  # Two get_cache hits
    assert stats["cache_misses"] == 1  # One get_cache miss
    assert stats["total_caches"] == 1
    print("  ✓ Statistics correct")

    print("✓ All basic operations passed!\n")


def test_lru_eviction() -> None:
    """Test LRU eviction policy."""
    print("Testing LRU eviction...")

    manager = KVCacheManager(device_type="cpu", device_id=0)

    # Create multiple caches
    print("  Creating multiple caches...")
    manager.create_cache(request_id="req-1", num_layers=12)
    time.sleep(0.01)
    manager.create_cache(request_id="req-2", num_layers=12)
    time.sleep(0.01)
    manager.create_cache(request_id="req-3", num_layers=12)

    # Mark all as complete
    print("  Marking requests as complete...")
    manager.mark_request_complete("req-1")
    manager.mark_request_complete("req-2")
    manager.mark_request_complete("req-3")

    # Access req-2 to make it most recent
    print("  Accessing req-2...")
    manager.get_cache("req-2")

    # Check LRU order
    print("  Verifying LRU order...")
    caches = list(manager._caches.items())
    inactive = [(rid, c) for rid, c in caches if rid not in manager._active_requests]
    inactive.sort(key=lambda x: x[1].last_accessed)

    assert inactive[0][0] == "req-1"  # Oldest
    assert inactive[1][0] == "req-3"  # Middle
    assert inactive[2][0] == "req-2"  # Newest
    print("  ✓ LRU order correct")

    # Test eviction
    print("  Testing eviction...")
    manager.evict_cache("req-1")
    assert manager.get_cache("req-1") is None
    assert manager.get_cache("req-2") is not None
    assert manager.get_cache("req-3") is not None
    print("  ✓ Eviction successful")

    stats = manager.get_stats()
    assert stats["total_evictions"] == 1
    print("  ✓ Eviction statistics correct")

    print("✓ LRU eviction tests passed!\n")


def test_active_request_protection() -> None:
    """Test that active requests are not evicted."""
    print("Testing active request protection...")

    manager = KVCacheManager(device_type="cpu", device_id=0)

    # Create caches
    print("  Creating active and inactive caches...")
    manager.create_cache(request_id="req-active", num_layers=12)
    manager.create_cache(request_id="req-inactive", num_layers=12)

    # Mark one as complete
    manager.mark_request_complete("req-inactive")

    # Check active status
    print("  Verifying active status...")
    assert "req-active" in manager._active_requests
    assert "req-inactive" not in manager._active_requests
    print("  ✓ Active status correct")

    # Verify inactive cache can be identified
    inactive = [rid for rid, _ in manager._caches.items() if rid not in manager._active_requests]
    assert "req-inactive" in inactive
    assert "req-active" not in inactive
    print("  ✓ Active request protected from eviction list")

    print("✓ Active request protection tests passed!\n")


def test_cache_statistics() -> None:
    """Test cache statistics tracking."""
    print("Testing cache statistics...")

    manager = KVCacheManager(device_type="cpu", device_id=0)

    # Create activity
    print("  Creating test activity...")
    manager.create_cache(request_id="req-1", num_layers=12)
    manager.create_cache(request_id="req-2", num_layers=12)
    manager.get_cache("req-1")  # Hit
    manager.get_cache("req-1")  # Hit
    manager.get_cache("req-3")  # Miss
    manager.evict_cache("req-2")

    # Check statistics
    print("  Verifying statistics...")
    stats = manager.get_stats()

    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 1
    assert stats["total_evictions"] == 1
    assert stats["total_caches"] == 1
    assert stats["active_caches"] == 1
    assert stats["active_requests"] == 1

    # Check hit rate calculation
    hit_rate = stats["hit_rate_percent"]
    expected_hit_rate = (2 / 3) * 100  # 2 hits out of 3 total accesses
    assert abs(hit_rate - expected_hit_rate) < 0.01  # type: ignore
    print(f"  ✓ Hit rate: {hit_rate:.1f}%")

    print("✓ Cache statistics tests passed!\n")


def test_clear_caches() -> None:
    """Test clearing all caches."""
    print("Testing cache clearing...")

    manager = KVCacheManager(device_type="cpu", device_id=0)

    # Create multiple caches
    print("  Creating caches...")
    manager.create_cache(request_id="req-1", num_layers=12)
    manager.create_cache(request_id="req-2", num_layers=12)
    manager.create_cache(request_id="req-3", num_layers=12)

    assert manager.get_cache_count() == 3
    print("  ✓ 3 caches created")

    # Clear all
    print("  Clearing all caches...")
    manager.clear_all_caches()

    assert manager.get_cache_count() == 0
    assert manager.get_active_request_count() == 0
    print("  ✓ All caches cleared")

    print("✓ Cache clearing tests passed!\n")


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("KV Cache Manager Test Suite")
    print("=" * 60)
    print()

    try:
        test_basic_operations()
        test_lru_eviction()
        test_active_request_protection()
        test_cache_statistics()
        test_clear_caches()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
