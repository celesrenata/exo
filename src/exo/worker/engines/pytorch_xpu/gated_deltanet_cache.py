"""Integration layer between GatedDeltaNetPersistentState and DynamicCache.

Provides ``GatedDeltaNetCache``, a cache wrapper that stores both standard KV
cache entries (for full-attention layers) and persistent GatedDeltaNet recurrent
state (for linear-attention layers). The wrapper is passed through the pipeline
via the ``past_key_values`` (plural) keyword argument — the same kwarg that
HuggingFace Qwen3.5 decoder layers expect.

CRITICAL: The kwarg name MUST remain ``past_key_values`` (plural). Using the
singular ``past_key_value`` causes the argument to be silently swallowed by
``**kwargs`` in the decoder layer forward signature, breaking recurrent state
propagation entirely.

Design goals:
    1. Store ``GatedDeltaNetPersistentState`` instances indexed by global layer index
    2. Delegate standard DynamicCache operations to the wrapped cache
    3. Provide a lookup method for GatedDeltaNet layers to retrieve their state
    4. Preserve full-attention layer behavior unchanged
    5. Support request lifecycle (reset, recycle) for continuous batching

**Validates: Requirements 5.1, 5.2, 5.5, 5.6, 5.7, 5.8**
"""

from __future__ import annotations

import logging
from typing import Any, final

from exo.worker.engines.pytorch_xpu.gated_deltanet_state import (
    GatedDeltaNetPersistentState,
)

logger = logging.getLogger(__name__)


@final
class GatedDeltaNetCache:
    """Cache wrapper integrating GatedDeltaNet persistent state with DynamicCache.

    This class wraps a HuggingFace ``DynamicCache`` instance and augments it with
    per-layer ``GatedDeltaNetPersistentState`` storage. It is designed to be passed
    as the ``past_key_values`` argument to decoder layers.

    Full-attention layers interact with the underlying ``DynamicCache`` transparently
    via attribute delegation. GatedDeltaNet (linear-attention) layers retrieve their
    persistent state by calling ``get_gated_deltanet_state(layer_index)``.

    The wrapper preserves the DynamicCache interface so that HuggingFace layer code
    that accesses cache methods (``get_seq_length``, ``update``, ``__getitem__``, etc.)
    continues to work without modification.

    Thread safety: NOT thread-safe. Designed for single-threaded per-rank decode.

    Usage in pipeline_parallel_shard.py:
        ```python
        # Create the cache (once per request):
        cache = GatedDeltaNetCache(dynamic_cache=DynamicCache())

        # Store persistent state for a GatedDeltaNet layer:
        cache.set_gated_deltanet_state(layer_index=3, state=persistent_state)

        # Pass to layers via past_key_values (PLURAL):
        layer_kwargs["past_key_values"] = cache

        # GatedDeltaNet layer retrieves its state:
        state = cache.get_gated_deltanet_state(layer_index=3)
        ```
    """

    def __init__(self, dynamic_cache: Any) -> None:
        """Initialize the cache wrapper.

        Args:
            dynamic_cache: A HuggingFace ``DynamicCache`` instance (or compatible).
                All standard cache operations are delegated to this object.
        """
        self._dynamic_cache: Any = dynamic_cache
        self._gated_deltanet_states: dict[int, GatedDeltaNetPersistentState] = {}

    @property
    def dynamic_cache(self) -> Any:
        """Access the underlying DynamicCache instance directly.

        Useful when code needs to interact with the raw HuggingFace cache
        without going through the delegation layer.
        """
        return self._dynamic_cache

    def get_gated_deltanet_state(
        self, layer_index: int
    ) -> GatedDeltaNetPersistentState | None:
        """Retrieve the persistent GatedDeltaNet state for a given layer.

        Args:
            layer_index: Global layer index (0-indexed across all model layers).

        Returns:
            The ``GatedDeltaNetPersistentState`` for the layer, or None if no
            persistent state has been registered for that layer.
        """
        return self._gated_deltanet_states.get(layer_index)

    def set_gated_deltanet_state(
        self, layer_index: int, state: GatedDeltaNetPersistentState
    ) -> None:
        """Register a persistent GatedDeltaNet state for a layer.

        Args:
            layer_index: Global layer index (0-indexed across all model layers).
            state: The persistent state container to store.

        Raises:
            ValueError: If the state's layer_index does not match the provided
                layer_index argument.
        """
        if state.layer_index != layer_index:
            raise ValueError(
                f"State layer_index ({state.layer_index}) does not match "
                f"provided layer_index ({layer_index})"
            )
        self._gated_deltanet_states[layer_index] = state
        logger.debug(
            "Registered GatedDeltaNet persistent state: layer=%d, request=%r",
            layer_index,
            state.request_identifier,
        )

    def has_gated_deltanet_state(self, layer_index: int) -> bool:
        """Check whether a persistent state exists for the given layer.

        Args:
            layer_index: Global layer index to check.

        Returns:
            True if a persistent state is registered for the layer.
        """
        return layer_index in self._gated_deltanet_states

    @property
    def gated_deltanet_layer_indices(self) -> list[int]:
        """Return sorted list of layer indices that have persistent state."""
        return sorted(self._gated_deltanet_states.keys())

    @property
    def gated_deltanet_state_count(self) -> int:
        """Return the number of registered GatedDeltaNet persistent states."""
        return len(self._gated_deltanet_states)

    def remove_gated_deltanet_state(self, layer_index: int) -> None:
        """Remove the persistent state for a layer (e.g., on request completion).

        Args:
            layer_index: Global layer index whose state should be removed.

        Does nothing if no state exists for the layer.
        """
        removed = self._gated_deltanet_states.pop(layer_index, None)
        if removed is not None:
            logger.debug(
                "Removed GatedDeltaNet persistent state: layer=%d, request=%r",
                layer_index,
                removed.request_identifier,
            )

    def reset_all_gated_deltanet_states(self) -> None:
        """Reset all registered GatedDeltaNet states (zeros tensors, keeps containers).

        Use this when restarting generation within the same request without
        deallocating the persistent state containers.
        """
        for state in self._gated_deltanet_states.values():
            state.reset()
        logger.debug(
            "Reset all GatedDeltaNet persistent states: count=%d",
            len(self._gated_deltanet_states),
        )

    def recycle_all_gated_deltanet_states(self) -> None:
        """Recycle all registered GatedDeltaNet states for reuse by new requests.

        Marks all containers as available and clears ownership. The containers
        remain in the registry for potential reuse via ``claim()``.
        """
        for state in self._gated_deltanet_states.values():
            state.recycle()
        logger.debug(
            "Recycled all GatedDeltaNet persistent states: count=%d",
            len(self._gated_deltanet_states),
        )

    def clear_gated_deltanet_states(self) -> None:
        """Remove all registered GatedDeltaNet states from the cache.

        Unlike ``recycle_all_gated_deltanet_states``, this drops references
        entirely. Use on request completion when the containers will not be reused.
        """
        count = len(self._gated_deltanet_states)
        self._gated_deltanet_states.clear()
        logger.debug(
            "Cleared all GatedDeltaNet persistent states: removed=%d", count
        )

    # -----------------------------------------------------------------------
    # DynamicCache interface delegation
    # -----------------------------------------------------------------------
    # The following methods and properties delegate to the underlying
    # DynamicCache so that HuggingFace layer code works transparently.

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying DynamicCache.

        This allows the wrapper to be used anywhere a DynamicCache is expected.
        Methods like ``get_seq_length()``, ``update()``, ``__getitem__()``, etc.
        are forwarded transparently.

        Note: ``__getattr__`` is only called when normal attribute lookup fails,
        so our own attributes (``_dynamic_cache``, ``_gated_deltanet_states``, etc.)
        are found first without delegation.
        """
        return getattr(self._dynamic_cache, name)

    def __len__(self) -> int:
        """Delegate len() to the underlying DynamicCache."""
        return len(self._dynamic_cache)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        """Delegate indexing to the underlying DynamicCache."""
        return self._dynamic_cache[index]

    def __iter__(self) -> Any:
        """Delegate iteration to the underlying DynamicCache."""
        return iter(self._dynamic_cache)

    def __repr__(self) -> str:
        """Informative representation showing both cache components."""
        return (
            f"GatedDeltaNetCache("
            f"dynamic_cache={self._dynamic_cache!r}, "
            f"gated_deltanet_states={len(self._gated_deltanet_states)} layers)"
        )
