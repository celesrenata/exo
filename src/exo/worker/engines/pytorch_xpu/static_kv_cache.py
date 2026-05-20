"""Pre-allocated static KV cache for zero-allocation decode on XPU.

Replaces HuggingFace ``DynamicCache`` for the optimized decode path. All tensors
are allocated once at generation start and reused across decode steps via in-place
slice assignment. The ``data_ptr()`` of each cache tensor remains constant for the
lifetime of the cache — no reallocation occurs during decode.

Tensor layout per full-attention layer:
    key_cache:   [1, max_seq_len, num_kv_heads, head_dim]
    value_cache: [1, max_seq_len, num_kv_heads, head_dim]

GatedDeltaNet layers store conv_state and recurrent_state in pre-allocated slots
rather than standard KV entries.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""

from __future__ import annotations

import logging
from typing import final

import torch

logger = logging.getLogger(__name__)


@final
class GatedDeltaNetStaticSlot:
    """Pre-allocated storage for a single GatedDeltaNet layer's state.

    Holds conv_state and recurrent_state tensors that persist across decode
    steps without reallocation. Updated in-place by the GatedDeltaNet layer.

    Attributes:
        conv_state: Causal conv1d sliding window, shape [1, conv_size, hidden_size].
            Stored in the model's compute dtype (bf16).
        recurrent_state: Recurrent state matrix, shape [1, num_heads, head_dim, head_dim].
            Stored in fp32 for numerical stability during state accumulation.
    """

    __slots__ = ("conv_state", "recurrent_state")

    def __init__(
        self,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state

    def reset(self) -> None:
        """Zero both state tensors without deallocating memory."""
        self.conv_state.zero_()
        self.recurrent_state.zero_()


@final
class StaticKVCache:
    """Pre-allocated KV cache for zero-allocation decode.

    All memory is allocated once during ``__init__`` and reused across decode
    steps. The position index advances by one per decode step (tracked as a
    Python int for slice indexing). After ``reset()``, the same tensor storage
    is reused for the next generation request.

    This class supports two layer types:
        - Full-attention layers: standard key/value cache tensors
        - GatedDeltaNet layers: conv_state and recurrent_state slots

    The ``update()`` method handles full-attention layers. GatedDeltaNet layers
    access their pre-allocated slots directly via ``get_gated_deltanet_slot()``.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    """

    __slots__ = (
        "_key_caches",
        "_value_caches",
        "_gated_deltanet_slots",
        "_position",
        "_num_attention_layers",
        "_num_kv_heads",
        "_head_dim",
        "_max_seq_len",
        "_device",
        "_dtype",
    )

    def __init__(
        self,
        num_attention_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        gated_deltanet_configs: list[GatedDeltaNetLayerConfig] | None = None,
    ) -> None:
        """Pre-allocate all cache tensors on the target device.

        Args:
            num_attention_layers: Number of full-attention layers requiring KV cache.
            num_kv_heads: Number of key/value heads per attention layer.
            head_dim: Dimension of each attention head.
            max_seq_len: Maximum sequence length to pre-allocate for.
            device: Target device for tensor allocation (e.g., torch.device("xpu:0")).
            dtype: Data type for KV cache tensors. Defaults to bfloat16.
            gated_deltanet_configs: Optional list of GatedDeltaNet layer configurations.
                Each entry describes the conv_state and recurrent_state shapes for one
                GatedDeltaNet layer. When provided, pre-allocates static slots for these
                layers.

        **Validates: Requirement 2.1**
        """
        self._num_attention_layers = num_attention_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._device = device
        self._dtype = dtype
        self._position: int = 0

        # Pre-allocate KV cache tensors for full-attention layers
        # Shape: [1, max_seq_len, num_kv_heads, head_dim]
        self._key_caches: list[torch.Tensor] = []
        self._value_caches: list[torch.Tensor] = []

        for _ in range(num_attention_layers):
            key_cache = torch.zeros(
                1, max_seq_len, num_kv_heads, head_dim,
                dtype=dtype,
                device=device,
            )
            value_cache = torch.zeros(
                1, max_seq_len, num_kv_heads, head_dim,
                dtype=dtype,
                device=device,
            )
            self._key_caches.append(key_cache)
            self._value_caches.append(value_cache)

        # Pre-allocate GatedDeltaNet state slots
        self._gated_deltanet_slots: list[GatedDeltaNetStaticSlot] = []

        if gated_deltanet_configs is not None:
            for config in gated_deltanet_configs:
                conv_state = torch.zeros(
                    1, config.conv_size, config.hidden_size,
                    dtype=dtype,
                    device=device,
                )
                recurrent_state = torch.zeros(
                    1, config.num_heads, config.head_dim, config.head_dim,
                    dtype=torch.float32,
                    device=device,
                )
                self._gated_deltanet_slots.append(
                    GatedDeltaNetStaticSlot(
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                    )
                )

        total_kv_bytes = (
            num_attention_layers * 2 * max_seq_len * num_kv_heads * head_dim
            * dtype_to_bytes(dtype)
        )
        gdn_bytes = sum(
            slot.conv_state.nelement() * dtype_to_bytes(dtype)
            + slot.recurrent_state.nelement() * 4  # fp32
            for slot in self._gated_deltanet_slots
        )

        logger.info(
            "StaticKVCache allocated: attention_layers=%d, gdn_layers=%d, "
            "max_seq_len=%d, num_kv_heads=%d, head_dim=%d, device=%s, "
            "kv_bytes=%d, gdn_bytes=%d, total_bytes=%d",
            num_attention_layers,
            len(self._gated_deltanet_slots),
            max_seq_len,
            num_kv_heads,
            head_dim,
            device,
            total_kv_bytes,
            gdn_bytes,
            total_kv_bytes + gdn_bytes,
        )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write K/V at current position, advance position, return full K/V up to position.

        Performs in-place slice assignment into the pre-allocated cache tensors.
        The ``data_ptr()`` of the underlying storage remains constant.

        Args:
            layer_idx: Index of the full-attention layer (0-indexed among attention layers).
            key: New key tensor, shape [1, 1, num_kv_heads, head_dim].
            value: New value tensor, shape [1, 1, num_kv_heads, head_dim].

        Returns:
            Tuple of (keys, values) sliced up to and including the current position.
            Shape: [1, position+1, num_kv_heads, head_dim] for each.

        Raises:
            IndexError: If layer_idx is out of range.
            RuntimeError: If position has reached max_seq_len.

        **Validates: Requirements 2.2, 2.3**
        """
        if layer_idx < 0 or layer_idx >= self._num_attention_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range "
                f"[0, {self._num_attention_layers})"
            )

        if self._position >= self._max_seq_len:
            raise RuntimeError(
                f"StaticKVCache position {self._position} has reached "
                f"max_seq_len {self._max_seq_len}"
            )

        pos = self._position

        # In-place slice assignment — no new tensor allocation
        # key shape: [1, 1, num_kv_heads, head_dim] -> write at position pos
        self._key_caches[layer_idx][:, pos:pos + 1, :, :] = key
        self._value_caches[layer_idx][:, pos:pos + 1, :, :] = value

        # Advance position only after the last layer updates
        # (caller is responsible for calling advance_position() after all layers)
        # Return the valid portion of the cache up to and including current position
        return (
            self._key_caches[layer_idx][:, :pos + 1, :, :],
            self._value_caches[layer_idx][:, :pos + 1, :, :],
        )

    def advance_position(self) -> None:
        """Advance the position index by one after all layers have been updated.

        Called once per decode step after all layers have written their K/V entries.

        **Validates: Requirement 2.3**
        """
        self._position += 1

    @property
    def position(self) -> int:
        """Current sequence position (number of decode steps completed).

        Advances by 1 per decode step. After N decode steps, position equals N.

        **Validates: Requirement 2.3**
        """
        return self._position

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length this cache was pre-allocated for."""
        return self._max_seq_len

    @property
    def num_attention_layers(self) -> int:
        """Number of full-attention layers with KV cache."""
        return self._num_attention_layers

    @property
    def num_gated_deltanet_layers(self) -> int:
        """Number of GatedDeltaNet layers with static state slots."""
        return len(self._gated_deltanet_slots)

    @property
    def device(self) -> torch.device:
        """Device where all cache tensors reside."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the KV cache tensors."""
        return self._dtype

    def get_key_cache(self, layer_idx: int) -> torch.Tensor:
        """Get the full pre-allocated key cache tensor for a layer.

        Returns the entire tensor (not sliced to position). Use for
        ``data_ptr()`` stability verification.

        Args:
            layer_idx: Index of the full-attention layer.

        Returns:
            Key cache tensor of shape [1, max_seq_len, num_kv_heads, head_dim].
        """
        return self._key_caches[layer_idx]

    def get_value_cache(self, layer_idx: int) -> torch.Tensor:
        """Get the full pre-allocated value cache tensor for a layer.

        Returns the entire tensor (not sliced to position). Use for
        ``data_ptr()`` stability verification.

        Args:
            layer_idx: Index of the full-attention layer.

        Returns:
            Value cache tensor of shape [1, max_seq_len, num_kv_heads, head_dim].
        """
        return self._value_caches[layer_idx]

    def get_gated_deltanet_slot(self, slot_idx: int) -> GatedDeltaNetStaticSlot:
        """Get the pre-allocated GatedDeltaNet state slot for a layer.

        Args:
            slot_idx: Index of the GatedDeltaNet layer (0-indexed among GDN layers).

        Returns:
            The static slot containing conv_state and recurrent_state tensors.

        Raises:
            IndexError: If slot_idx is out of range.

        **Validates: Requirement 2.5**
        """
        if slot_idx < 0 or slot_idx >= len(self._gated_deltanet_slots):
            raise IndexError(
                f"slot_idx {slot_idx} out of range "
                f"[0, {len(self._gated_deltanet_slots)})"
            )
        return self._gated_deltanet_slots[slot_idx]

    def reset(self) -> None:
        """Reset position to 0 for next request without deallocating memory.

        Zeros all cache tensors and resets the position counter. The underlying
        tensor storage (``data_ptr()``) remains unchanged — no deallocation or
        reallocation occurs.

        **Validates: Requirement 2.4**
        """
        self._position = 0

        for key_cache in self._key_caches:
            key_cache.zero_()
        for value_cache in self._value_caches:
            value_cache.zero_()

        for slot in self._gated_deltanet_slots:
            slot.reset()

        logger.debug(
            "StaticKVCache reset: position=0, attention_layers=%d, "
            "gdn_layers=%d, device=%s",
            self._num_attention_layers,
            len(self._gated_deltanet_slots),
            self._device,
        )


@final
class GatedDeltaNetLayerConfig:
    """Configuration for a single GatedDeltaNet layer's static state allocation.

    Describes the shapes needed to pre-allocate conv_state and recurrent_state
    tensors for one GatedDeltaNet layer.
    """

    __slots__ = ("conv_size", "hidden_size", "num_heads", "head_dim")

    def __init__(
        self,
        conv_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize GatedDeltaNet layer configuration.

        Args:
            conv_size: Kernel size of the causal conv1d (typically 4).
            hidden_size: Input dimension of the conv1d (typically num_heads * (key_dim + value_dim)).
            num_heads: Number of attention heads for the recurrent state.
            head_dim: Dimension per head for the recurrent state matrix.
        """
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim


def dtype_to_bytes(dtype: torch.dtype) -> int:
    """Return the number of bytes per element for a given dtype.

    Args:
        dtype: PyTorch data type.

    Returns:
        Number of bytes per element.
    """
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.int8 or dtype == torch.uint8:
        return 1
    elif dtype == torch.int16:
        return 2
    elif dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    else:
        # Fallback: use torch's element_size
        return torch.tensor([], dtype=dtype).element_size()
