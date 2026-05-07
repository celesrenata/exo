"""KV cache allocation and management on GPU devices.

Manages key and value cache tensors for transformer attention layers.
Supports pre-allocation on the target GPU device and dynamic sequence
length growth when positions exceed the current allocation.

Uses device-agnostic APIs (torch.zeros with device parameter, tensor.to(device))
for all tensor operations.

Requirements: 2.6, 2.8
"""

from __future__ import annotations

import logging
from typing import final

logger = logging.getLogger(__name__)


@final
class KVCache:
    """Manages key and value cache tensors for transformer attention.

    Pre-allocates cache tensors on the target GPU device and supports
    dynamic growth when sequence length exceeds the current allocation.

    Attributes:
        num_layers: Number of transformer layers to cache.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each attention head.
        max_seq_len: Current maximum sequence length allocation.
        device: Target device string (e.g. "xpu:0", "cuda:0").
        dtype: Tensor data type for cache storage.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: str,
        dtype: "torch.dtype | None" = None,
    ) -> None:
        """Initialize KV cache configuration.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads per layer.
            head_dim: Dimension of each attention head.
            max_seq_len: Initial maximum sequence length to pre-allocate.
            device: Target device string (e.g. "xpu:0", "cuda:0").
            dtype: Tensor dtype for cache. Defaults to torch.float16.
        """
        import torch

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float16

        # Cache tensors: list of (key_cache, value_cache) per layer
        self._key_caches: list[torch.Tensor] = []
        self._value_caches: list[torch.Tensor] = []
        self._current_seq_len: int = 0
        self._allocated: bool = False

    @property
    def current_seq_len(self) -> int:
        """Return the current sequence length stored in the cache."""
        return self._current_seq_len

    def allocate(self) -> None:
        """Pre-allocate cache tensors on the target device.

        Creates zero-filled tensors of shape (num_heads, max_seq_len, head_dim)
        for both keys and values at each layer. Uses torch.zeros with the
        target device for device-agnostic allocation.
        """
        import torch

        self._key_caches = []
        self._value_caches = []

        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                self.num_heads,
                self.max_seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            value_cache = torch.zeros(
                self.num_heads,
                self.max_seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self._key_caches.append(key_cache)
            self._value_caches.append(value_cache)

        self._current_seq_len = 0
        self._allocated = True

        logger.info(
            "KV cache allocated: %d layers, %d heads, head_dim=%d, "
            "max_seq_len=%d on %s (%s)",
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.max_seq_len,
            self.device,
            self.dtype,
        )

    def _grow(self, new_max_seq_len: int) -> None:
        """Grow the cache to accommodate a larger sequence length.

        Allocates new larger tensors, copies existing data, and replaces
        the old cache tensors. The new allocation is at least double the
        current size or the requested size, whichever is larger.

        Args:
            new_max_seq_len: Minimum required sequence length.
        """
        import torch

        # Grow to at least double or the requested size
        new_size = max(new_max_seq_len, self.max_seq_len * 2)

        logger.info(
            "Growing KV cache from max_seq_len=%d to %d on %s",
            self.max_seq_len,
            new_size,
            self.device,
        )

        new_key_caches: list[torch.Tensor] = []
        new_value_caches: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            # Allocate new tensors
            new_key = torch.zeros(
                self.num_heads,
                new_size,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            new_value = torch.zeros(
                self.num_heads,
                new_size,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            # Copy existing data
            if self._current_seq_len > 0:
                new_key[:, : self._current_seq_len, :] = self._key_caches[layer_idx][
                    :, : self._current_seq_len, :
                ]
                new_value[:, : self._current_seq_len, :] = self._value_caches[layer_idx][
                    :, : self._current_seq_len, :
                ]

            new_key_caches.append(new_key)
            new_value_caches.append(new_value)

        self._key_caches = new_key_caches
        self._value_caches = new_value_caches
        self.max_seq_len = new_size

    def update(
        self,
        layer_idx: int,
        key: "torch.Tensor",
        value: "torch.Tensor",
        position: int,
    ) -> None:
        """Update the cache at the given position for a specific layer.

        If the position exceeds the current allocation, the cache is
        dynamically grown to accommodate the new position.

        Args:
            layer_idx: Transformer layer index (0-based).
            key: Key tensor of shape (num_heads, seq_len, head_dim) or
                (num_heads, 1, head_dim) for single-token updates.
            value: Value tensor of shape matching key.
            position: Starting position in the sequence to write at.

        Raises:
            RuntimeError: If cache has not been allocated.
            IndexError: If layer_idx is out of range.
        """
        if not self._allocated:
            raise RuntimeError(
                "KV cache has not been allocated. Call allocate() first."
            )

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )

        # Determine how many positions we're writing
        seq_len = key.shape[-2]  # (num_heads, seq_len, head_dim)
        end_position = position + seq_len

        # Grow if needed
        if end_position > self.max_seq_len:
            self._grow(end_position)

        # Move tensors to cache device if needed
        key_on_device = key.to(self.device)
        value_on_device = value.to(self.device)

        # Write into cache
        self._key_caches[layer_idx][:, position:end_position, :] = key_on_device
        self._value_caches[layer_idx][:, position:end_position, :] = value_on_device

        # Update current sequence length
        if end_position > self._current_seq_len:
            self._current_seq_len = end_position

    def get(
        self,
        layer_idx: int,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return cached key/value tensors up to seq_len for a layer.

        Args:
            layer_idx: Transformer layer index (0-based).
            seq_len: Number of positions to return. If None, returns up to
                current_seq_len.

        Returns:
            Tuple of (key_cache, value_cache) each of shape
            (num_heads, seq_len, head_dim).

        Raises:
            RuntimeError: If cache has not been allocated.
            IndexError: If layer_idx is out of range.
        """
        if not self._allocated:
            raise RuntimeError(
                "KV cache has not been allocated. Call allocate() first."
            )

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )

        length = seq_len if seq_len is not None else self._current_seq_len

        return (
            self._key_caches[layer_idx][:, :length, :],
            self._value_caches[layer_idx][:, :length, :],
        )

    def clear(self) -> None:
        """Reset the cache, zeroing all stored values.

        Keeps the allocated tensors but resets the sequence length counter
        and fills tensors with zeros. This avoids re-allocation overhead
        when starting a new generation.
        """
        if not self._allocated:
            return

        for layer_idx in range(self.num_layers):
            self._key_caches[layer_idx].zero_()
            self._value_caches[layer_idx].zero_()

        self._current_seq_len = 0
        logger.debug("KV cache cleared (allocation preserved)")


# Type import for annotations only
import typing

if typing.TYPE_CHECKING:
    import torch
