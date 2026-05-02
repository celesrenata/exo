"""
KV Cache Manager for PyTorch XPU Backend

This module provides key-value cache management for transformer inference,
including LRU eviction, memory monitoring, and cache statistics.

Requirements addressed:
- 4.1: KV cache data structures and management
- 4.2: LRU eviction policy
- 4.3: Memory monitoring and limits
- 4.4: Cache hit/miss tracking
- 4.5: Cache statistics
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, final

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVCache:
    """
    Key-Value cache for a single inference request.

    This stores the cached key and value tensors for each transformer layer,
    enabling efficient incremental inference without recomputing past tokens.

    Requirements: 4.1
    """

    request_id: str
    keys: list[Any]  # List of key tensors, one per layer
    values: list[Any]  # List of value tensors, one per layer
    position: int  # Current position in the sequence
    max_length: int  # Maximum sequence length
    last_accessed: float = field(default_factory=time.time)  # Last access timestamp
    is_active: bool = True  # Whether this cache is for an active request


@dataclass
class CacheStatistics:
    """
    Statistics for cache performance monitoring.

    Requirements: 4.5, 9.2
    """

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_evictions: int = 0
    active_caches: int = 0
    total_memory_bytes: int = 0
    peak_memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as a percentage."""
        return 100.0 - self.hit_rate


@final
class KVCacheManager:
    """
    Manages key-value caches for transformer inference.

    This class provides:
    - Per-request KV cache allocation and management
    - LRU eviction when memory is constrained
    - Memory monitoring and threshold enforcement
    - Cache statistics for performance monitoring

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """

    def __init__(
        self,
        device_type: str,
        device_id: int,
        memory_threshold_percent: float = 80.0,
    ) -> None:
        """
        Initialize the KVCacheManager.

        Args:
            device_type: Device type string ("xpu", "cuda", or "cpu")
            device_id: Device ID
            memory_threshold_percent: Percentage of GPU memory to use before eviction (default: 80%)

        Requirements: 4.1
        """
        self._device_type = device_type
        self._device_id = device_id
        self._memory_threshold_percent = memory_threshold_percent

        # Cache storage: request_id -> KVCache
        self._caches: dict[str, KVCache] = {}

        # Statistics
        self._stats = CacheStatistics()

        # Track active requests (requests currently being processed)
        self._active_requests: set[str] = set()

        # Try to import torch for memory monitoring
        self._torch_available = False
        try:
            import torch

            self._torch_available = True
            self._torch = torch
        except ImportError:
            logger.warning("PyTorch not available - memory monitoring disabled")

        logger.info(
            f"KVCacheManager initialized for {device_type}:{device_id}, "
            f"memory threshold: {memory_threshold_percent}%"
        )

    def get_cache(self, request_id: str) -> Optional[KVCache]:
        """
        Get the KV cache for a request.

        Args:
            request_id: Unique identifier for the request

        Returns:
            KVCache if it exists, None otherwise

        Requirements: 4.1, 4.4
        """
        if request_id in self._caches:
            # Update last accessed time
            cache = self._caches[request_id]
            # Create new cache with updated timestamp (frozen dataclass)
            updated_cache = KVCache(
                request_id=cache.request_id,
                keys=cache.keys,
                values=cache.values,
                position=cache.position,
                max_length=cache.max_length,
                last_accessed=time.time(),
                is_active=cache.is_active,
            )
            self._caches[request_id] = updated_cache

            self._stats.cache_hits += 1
            logger.debug(f"Cache hit for request {request_id}")
            return updated_cache
        else:
            self._stats.cache_misses += 1
            logger.debug(f"Cache miss for request {request_id}")
            return None

    def create_cache(
        self,
        request_id: str,
        num_layers: int,
        max_length: int = 8192,
    ) -> KVCache:
        """
        Create a new KV cache for a request.

        Args:
            request_id: Unique identifier for the request
            num_layers: Number of transformer layers
            max_length: Maximum sequence length (default: 8192)

        Returns:
            Newly created KVCache

        Raises:
            ValueError: If cache already exists for this request

        Requirements: 4.1
        """
        if request_id in self._caches:
            raise ValueError(f"Cache already exists for request {request_id}")

        # Create empty lists for keys and values (one per layer)
        keys: list[Any] = [None] * num_layers
        values: list[Any] = [None] * num_layers

        # Create cache
        cache = KVCache(
            request_id=request_id,
            keys=keys,
            values=values,
            position=0,
            max_length=max_length,
            last_accessed=time.time(),
            is_active=True,
        )

        # Store cache
        self._caches[request_id] = cache
        self._active_requests.add(request_id)
        self._stats.total_requests += 1
        self._stats.active_caches += 1

        logger.info(
            f"Created cache for request {request_id}: "
            f"{num_layers} layers, max_length={max_length}"
        )

        # Check if we need to evict old caches
        self._check_memory_and_evict()

        return cache

    def update_cache(
        self,
        request_id: str,
        layer_idx: int,
        new_key: object,
        new_value: object,
        new_position: Optional[int] = None,
    ) -> None:
        """
        Update the KV cache for a specific layer.

        Args:
            request_id: Unique identifier for the request
            layer_idx: Index of the layer to update
            new_key: New key tensor to cache
            new_value: New value tensor to cache
            new_position: Optional new position in sequence

        Raises:
            ValueError: If cache doesn't exist for this request

        Requirements: 4.1
        """
        if request_id not in self._caches:
            raise ValueError(f"No cache exists for request {request_id}")

        cache = self._caches[request_id]

        # Update keys and values (need to create new lists since frozen)
        new_keys = cache.keys.copy()
        new_values = cache.values.copy()
        new_keys[layer_idx] = new_key
        new_values[layer_idx] = new_value

        # Determine new position
        position = new_position if new_position is not None else cache.position + 1

        # Create updated cache
        updated_cache = KVCache(
            request_id=cache.request_id,
            keys=new_keys,
            values=new_values,
            position=position,
            max_length=cache.max_length,
            last_accessed=time.time(),
            is_active=cache.is_active,
        )

        self._caches[request_id] = updated_cache

        logger.debug(
            f"Updated cache for request {request_id}, layer {layer_idx}, position {position}"
        )

    def evict_cache(self, request_id: str) -> None:
        """
        Evict (remove) the cache for a request.

        Args:
            request_id: Unique identifier for the request

        Requirements: 4.1, 4.2
        """
        if request_id not in self._caches:
            logger.warning(f"Attempted to evict non-existent cache for request {request_id}")
            return

        # Remove from caches
        del self._caches[request_id]

        # Remove from active requests if present
        self._active_requests.discard(request_id)

        # Update statistics
        self._stats.total_evictions += 1
        self._stats.active_caches = len(self._caches)

        logger.info(f"Evicted cache for request {request_id}")

    def mark_request_complete(self, request_id: str) -> None:
        """
        Mark a request as complete (no longer active).

        This allows the cache to be evicted by LRU policy if needed.

        Args:
            request_id: Unique identifier for the request

        Requirements: 4.2
        """
        if request_id in self._active_requests:
            self._active_requests.remove(request_id)

            # Update cache to mark as inactive
            if request_id in self._caches:
                cache = self._caches[request_id]
                updated_cache = KVCache(
                    request_id=cache.request_id,
                    keys=cache.keys,
                    values=cache.values,
                    position=cache.position,
                    max_length=cache.max_length,
                    last_accessed=cache.last_accessed,
                    is_active=False,
                )
                self._caches[request_id] = updated_cache

            logger.debug(f"Marked request {request_id} as complete")

    def _check_memory_and_evict(self) -> None:
        """
        Check memory usage and evict old caches if threshold exceeded.

        Uses LRU (Least Recently Used) policy to evict caches.
        Never evicts caches for active requests.

        Requirements: 4.2, 4.3
        """
        if not self._torch_available:
            return

        # Get current memory usage
        total_memory, free_memory = self._get_device_memory()
        if total_memory == 0:
            return  # CPU or memory info unavailable

        used_memory = total_memory - free_memory
        usage_percent = (used_memory / total_memory) * 100.0

        # Update statistics
        self._stats.total_memory_bytes = used_memory
        if used_memory > self._stats.peak_memory_bytes:
            self._stats.peak_memory_bytes = used_memory

        # Check if we need to evict
        if usage_percent < self._memory_threshold_percent:
            return

        logger.info(
            f"Memory usage {usage_percent:.1f}% exceeds threshold "
            f"{self._memory_threshold_percent}%, triggering eviction"
        )

        # Get inactive caches sorted by last access time (oldest first)
        inactive_caches = [
            (request_id, cache)
            for request_id, cache in self._caches.items()
            if request_id not in self._active_requests
        ]

        if not inactive_caches:
            logger.warning(
                "Memory threshold exceeded but no inactive caches to evict. "
                "All caches are for active requests."
            )
            return

        # Sort by last accessed time (oldest first)
        inactive_caches.sort(key=lambda x: x[1].last_accessed)

        # Evict oldest caches until we're below threshold
        evicted_count = 0
        for request_id, _ in inactive_caches:
            self.evict_cache(request_id)
            evicted_count += 1

            # Check memory again
            total_memory, free_memory = self._get_device_memory()
            used_memory = total_memory - free_memory
            usage_percent = (used_memory / total_memory) * 100.0

            if usage_percent < self._memory_threshold_percent:
                break

        logger.info(
            f"Evicted {evicted_count} cache(s), "
            f"new memory usage: {usage_percent:.1f}%"
        )

    def _get_device_memory(self) -> tuple[int, int]:
        """
        Get device memory information.

        Returns:
            Tuple of (total_memory_bytes, free_memory_bytes)

        Requirements: 4.3, 9.2
        """
        if not self._torch_available:
            return (0, 0)

        try:
            if self._device_type == "xpu" and hasattr(self._torch, "xpu"):
                props = self._torch.xpu.get_device_properties(self._device_id)  # type: ignore
                allocated = self._torch.xpu.memory_allocated(self._device_id)  # type: ignore
                total_mem: int = props.total_memory  # type: ignore
                free_mem: int = total_mem - allocated
                return (total_mem, free_mem)

            elif self._device_type == "cuda":
                props = self._torch.cuda.get_device_properties(self._device_id)  # type: ignore
                allocated = self._torch.cuda.memory_allocated(self._device_id)  # type: ignore
                total_mem: int = props.total_memory  # type: ignore
                free_mem: int = total_mem - allocated
                return (total_mem, free_mem)

            else:
                # CPU has no meaningful memory limit
                return (0, 0)

        except Exception as e:
            logger.error(f"Error getting device memory: {e}")
            return (0, 0)

    def get_stats(self) -> dict[str, object]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics including:
            - total_requests: Total number of requests processed
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate percentage
            - miss_rate: Cache miss rate percentage
            - total_evictions: Total number of evictions
            - active_caches: Number of currently active caches
            - total_memory_bytes: Current total memory usage
            - peak_memory_bytes: Peak memory usage

        Requirements: 4.5, 9.2
        """
        stats: dict[str, object] = {
            "total_requests": self._stats.total_requests,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate_percent": self._stats.hit_rate,
            "miss_rate_percent": self._stats.miss_rate,
            "total_evictions": self._stats.total_evictions,
            "active_caches": self._stats.active_caches,
            "total_caches": len(self._caches),
            "active_requests": len(self._active_requests),
            "total_memory_bytes": self._stats.total_memory_bytes,
            "peak_memory_bytes": self._stats.peak_memory_bytes,
        }

        # Add memory info if available
        if self._torch_available:
            total_memory, free_memory = self._get_device_memory()
            if total_memory > 0:
                used_memory = total_memory - free_memory
                stats["device_total_memory_bytes"] = total_memory
                stats["device_free_memory_bytes"] = free_memory
                stats["device_used_memory_bytes"] = used_memory
                stats["device_memory_utilization_percent"] = (
                    used_memory / total_memory * 100.0
                )

        logger.debug(f"Cache statistics: {stats}")
        return stats

    def clear_all_caches(self) -> None:
        """
        Clear all caches.

        This is useful for testing or when resetting the system.

        Requirements: 4.1
        """
        num_caches = len(self._caches)
        self._caches.clear()
        self._active_requests.clear()
        self._stats.active_caches = 0

        logger.info(f"Cleared all {num_caches} cache(s)")

    def get_cache_count(self) -> int:
        """
        Get the number of active caches.

        Returns:
            Number of caches currently stored

        Requirements: 4.5
        """
        return len(self._caches)

    def get_active_request_count(self) -> int:
        """
        Get the number of active requests.

        Returns:
            Number of requests currently being processed

        Requirements: 4.5
        """
        return len(self._active_requests)
