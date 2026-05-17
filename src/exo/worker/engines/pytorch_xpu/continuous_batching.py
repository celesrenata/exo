"""
Continuous batching scheduler for concurrent request processing.

Manages request lifecycle through admission, prefill, decode, completion,
and cancellation stages. The scheduler tracks which requests are active and
determines which requests should participate in the next decode microbatch.

This module is NOT thread-safe — it assumes single-threaded per-rank execution.
It does NOT manage actual inference; it only tracks request lifecycle and
determines batch composition.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.9, 3.10, 3.11
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, final

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.gated_deltanet_cache import (
        GatedDeltaNetCache,
    )
    from exo.worker.engines.pytorch_xpu.gated_deltanet_state import (
        GatedDeltaNetPersistentState,
    )
    from exo.worker.engines.pytorch_xpu.sampling import SamplingConfiguration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RequestState — lifecycle state for a single request
# ---------------------------------------------------------------------------

RequestState = Literal[
    "admitted",
    "prefilling",
    "decoding",
    "completed",
    "cancelled",
]
"""
Lifecycle state for a request managed by the scheduler.

- ``"admitted"``: In admission queue, waiting for prefill slot.
- ``"prefilling"``: Currently being prefilled (one at a time).
- ``"decoding"``: In decode-ready queue, actively generating tokens.
- ``"completed"``: Finished generation normally.
- ``"cancelled"``: Cancelled by user or system.
"""


# ---------------------------------------------------------------------------
# RequestRuntimeState — mutable runtime state for an active request
# ---------------------------------------------------------------------------


@final
@dataclass
class RequestRuntimeState:
    """
    Mutable runtime state for an active request during decode.

    Tracks the current position in the decode microbatch, token generation
    progress, and sampling configuration. Updated in place on each decode step.

    The ``slot_generation`` counter is incremented whenever a batch slot is
    reused for a new request, preventing stale in-flight messages from
    corrupting a reused slot.
    """

    request_id: str
    """Globally unique identifier for this request."""

    slot_index: int
    """Position in the decode microbatch."""

    slot_generation: int
    """Incremented on slot reuse to prevent stale messages."""

    tokens_generated: int
    """Count of tokens generated so far."""

    last_token_id: int
    """Most recently generated token."""

    is_finished: bool
    """Whether generation is complete (EOS or max_tokens)."""

    max_tokens: int
    """Maximum tokens to generate for this request."""

    sampling_config: SamplingConfiguration
    """Per-request sampling parameters."""


# ---------------------------------------------------------------------------
# BatchSlotState — state of a single slot in the decode microbatch
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class BatchSlotState:
    """
    Immutable state of a single slot in the decode microbatch.

    Represents whether a slot is occupied by an active request or empty
    (available for assignment). The ``slot_generation`` counter allows
    detection of stale messages targeting a previously occupied slot.
    """

    slot_index: int
    """Position of this slot in the microbatch."""

    request_id: str | None
    """Request occupying this slot, or None if the slot is empty."""

    slot_generation: int
    """Generation counter for staleness detection."""

    is_active: bool
    """Whether this slot is actively participating in decode."""


# ---------------------------------------------------------------------------
# DecodeMicrobatch — describes a decode step's batch composition
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class DecodeMicrobatch:
    """
    Immutable description of a decode step's batch composition.

    Contains the slot states and input tokens for all active slots in a
    single decode step. Used to communicate batch layout between the
    scheduler and the pipeline stages.
    """

    active_slot_count: int
    """Number of active slots in this step."""

    max_batch_size: int
    """Maximum batch capacity (total number of slots)."""

    slot_states: tuple[BatchSlotState, ...]
    """State of each slot in the microbatch."""

    input_token_ids: tuple[int, ...]
    """Last token for each active slot (length equals active_slot_count)."""


# ---------------------------------------------------------------------------
# TokenResultBatch — results from a decode step
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class TokenResultBatch:
    """
    Immutable results from a decode step.

    Contains the generated tokens and their corresponding slot metadata.
    The ``slot_generations`` tuple allows receivers to detect and discard
    stale results from slots that have been reassigned.

    All tuples must have the same length, equal to the number of active
    slots that produced results in this decode step.
    """

    token_ids: tuple[int, ...]
    """Generated token for each active slot."""

    slot_indices: tuple[int, ...]
    """Which slots these results correspond to."""

    slot_generations: tuple[int, ...]
    """Generation counters for staleness check."""


# ---------------------------------------------------------------------------
# SchedulerMetrics — frozen snapshot of scheduler state
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class SchedulerMetrics:
    """Frozen snapshot of scheduler state for instrumentation and diagnostics."""

    active_requests: int
    """Total number of requests that are admitted, prefilling, or decoding."""

    admission_queue_size: int
    """Number of requests waiting in the admission queue."""

    prefilling_count: int
    """Number of requests currently being prefilled (0 or 1)."""

    decode_ready_count: int
    """Number of requests ready for the next decode step."""

    completed_count: int
    """Total number of requests that have completed generation."""

    cancelled_count: int
    """Total number of requests that have been cancelled."""


# ---------------------------------------------------------------------------
# _RequestEntry — internal bookkeeping for a single request
# ---------------------------------------------------------------------------


@final
@dataclass
class _RequestEntry:
    """Internal mutable bookkeeping for a request in the scheduler."""

    request_id: str
    prompt_tokens: list[int]
    sampling_config: SamplingConfiguration
    state: RequestState


# ---------------------------------------------------------------------------
# ContinuousBatchScheduler — main scheduler class
# ---------------------------------------------------------------------------


@final
class ContinuousBatchScheduler:
    """
    Manages request lifecycle for continuous batching.

    Tracks requests through admission → prefill → decode → completion/cancellation.
    Forms decode microbatches from all decode-ready requests.

    This scheduler is single-threaded and does not perform inference. It only
    determines which requests should participate in the next decode step.
    """

    def __init__(self) -> None:
        """Initialize an empty scheduler with no active requests."""
        # Ordered queue of request IDs waiting for prefill
        self._admission_queue: deque[str] = deque()

        # Request currently being prefilled (at most one)
        self._prefilling: str | None = None

        # Set of request IDs ready for decode batching
        self._decode_ready: set[str] = set()

        # Tracking sets for terminal states
        self._completed: set[str] = set()
        self._cancelled: set[str] = set()

        # Master lookup: request_id → entry (only for active requests)
        self._entries: dict[str, _RequestEntry] = {}

    # -------------------------------------------------------------------
    # Public API: Request admission
    # -------------------------------------------------------------------

    def admit_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        sampling_config: SamplingConfiguration,
    ) -> None:
        """Add a new request to the admission queue.

        Args:
            request_id: Globally unique identifier for this request.
            prompt_tokens: Token IDs comprising the prompt.
            sampling_config: Sampling parameters for this request.

        Raises:
            ValueError: If request_id is already known to the scheduler
                (active, completed, or cancelled).
        """
        if request_id in self._entries:
            raise ValueError(
                f"Request '{request_id}' is already active in the scheduler"
            )
        if request_id in self._completed:
            raise ValueError(
                f"Request '{request_id}' has already completed"
            )
        if request_id in self._cancelled:
            raise ValueError(
                f"Request '{request_id}' has already been cancelled"
            )

        entry = _RequestEntry(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_config=sampling_config,
            state="admitted",
        )
        self._entries[request_id] = entry
        self._admission_queue.append(request_id)

    # -------------------------------------------------------------------
    # Public API: Prefill management
    # -------------------------------------------------------------------

    def get_next_prefill(self) -> str | None:
        """Pop the next request from the admission queue for prefill.

        Moves the request from "admitted" to "prefilling" state. Only one
        request can be prefilling at a time (sequential prefill).

        Returns:
            The request ID to prefill, or None if the admission queue is
            empty or a prefill is already in progress.
        """
        if self._prefilling is not None:
            return None
        if not self._admission_queue:
            return None

        request_id = self._admission_queue.popleft()
        entry = self._entries[request_id]
        entry.state = "prefilling"
        self._prefilling = request_id
        return request_id

    def mark_prefill_complete(self, request_id: str) -> None:
        """Move a request from prefilling to decode-ready state.

        Args:
            request_id: The request that has completed prefill.

        Raises:
            ValueError: If the request is not currently prefilling.
        """
        if self._prefilling != request_id:
            raise ValueError(
                f"Request '{request_id}' is not currently prefilling "
                f"(current prefilling: {self._prefilling})"
            )

        entry = self._entries[request_id]
        entry.state = "decoding"
        self._decode_ready.add(request_id)
        self._prefilling = None

    # -------------------------------------------------------------------
    # Public API: Decode batch formation
    # -------------------------------------------------------------------

    def get_decode_batch(self) -> list[str]:
        """Get all decode-ready request IDs for the next decode step.

        Returns:
            List of request IDs that should participate in the next decode
            microbatch. Order is not guaranteed to be stable across calls.
        """
        return list(self._decode_ready)

    # -------------------------------------------------------------------
    # Public API: Completion and cancellation
    # -------------------------------------------------------------------

    def mark_completed(self, request_id: str) -> None:
        """Move a request to the completed set.

        The request must be in an active state (admitted, prefilling, or
        decoding). After completion, the request is removed from all active
        tracking structures.

        Args:
            request_id: The request that has finished generation.

        Raises:
            ValueError: If the request is not active.
        """
        if request_id not in self._entries:
            raise ValueError(
                f"Request '{request_id}' is not active in the scheduler"
            )

        entry = self._entries[request_id]
        self._remove_from_active_structures(request_id, entry.state)
        del self._entries[request_id]
        self._completed.add(request_id)

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request and remove it from all active queues.

        The request must be in an active state (admitted, prefilling, or
        decoding). After cancellation, the request is removed from all
        active tracking structures.

        Args:
            request_id: The request to cancel.

        Raises:
            ValueError: If the request is not active.
        """
        if request_id not in self._entries:
            raise ValueError(
                f"Request '{request_id}' is not active in the scheduler"
            )

        entry = self._entries[request_id]
        self._remove_from_active_structures(request_id, entry.state)
        del self._entries[request_id]
        self._cancelled.add(request_id)

    # -------------------------------------------------------------------
    # Public API: Query methods
    # -------------------------------------------------------------------

    def is_active(self, request_id: str) -> bool:
        """Whether a request is still active (not completed or cancelled).

        Returns:
            True if the request is in admitted, prefilling, or decoding state.
        """
        return request_id in self._entries

    def get_request_state(self, request_id: str) -> RequestState | None:
        """Get the current state of a request.

        Returns:
            The request's lifecycle state, or None if the request is unknown.
        """
        entry = self._entries.get(request_id)
        if entry is not None:
            return entry.state
        if request_id in self._completed:
            return "completed"
        if request_id in self._cancelled:
            return "cancelled"
        return None

    def get_sampling_config(
        self, request_id: str
    ) -> SamplingConfiguration | None:
        """Get the sampling configuration for an active request.

        Returns:
            The sampling configuration, or None if the request is not active.
        """
        entry = self._entries.get(request_id)
        if entry is not None:
            return entry.sampling_config
        return None

    def get_metrics(self) -> SchedulerMetrics:
        """Create a frozen snapshot of the current scheduler state."""
        return SchedulerMetrics(
            active_requests=len(self._entries),
            admission_queue_size=len(self._admission_queue),
            prefilling_count=1 if self._prefilling is not None else 0,
            decode_ready_count=len(self._decode_ready),
            completed_count=len(self._completed),
            cancelled_count=len(self._cancelled),
        )

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def active_request_count(self) -> int:
        """Total number of active requests (admitted + prefilling + decoding)."""
        return len(self._entries)

    @property
    def decode_ready_count(self) -> int:
        """Number of requests ready for the next decode step."""
        return len(self._decode_ready)

    @property
    def admission_queue_size(self) -> int:
        """Number of requests waiting in the admission queue."""
        return len(self._admission_queue)

    @property
    def completed_count(self) -> int:
        """Total number of completed requests."""
        return len(self._completed)

    @property
    def cancelled_count(self) -> int:
        """Total number of cancelled requests."""
        return len(self._cancelled)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _remove_from_active_structures(
        self, request_id: str, state: RequestState
    ) -> None:
        """Remove a request from whichever active structure it belongs to."""
        if state == "admitted":
            # Remove from admission queue (deque doesn't have O(1) remove,
            # but admission queues are expected to be short)
            try:
                self._admission_queue.remove(request_id)
            except ValueError:
                pass
        elif state == "prefilling":
            if self._prefilling == request_id:
                self._prefilling = None
        elif state == "decoding":
            self._decode_ready.discard(request_id)


# ---------------------------------------------------------------------------
# RequestIdentifierMap — bidirectional string ↔ numeric slot mapping
# ---------------------------------------------------------------------------


@final
class RequestIdentifierMap:
    """Bidirectional mapping between string request IDs and numeric slot indices.

    Used for compact distributed packet encoding. String request IDs are
    mapped to small integers (slot indices) for efficient wire format.
    The reverse mapping is maintained on rank 0 for result routing.

    Includes slot generation counters to detect and discard stale messages
    from slots that have been reassigned to new requests.
    """

    def __init__(self, max_batch_size: int) -> None:
        """Initialize with a fixed maximum batch size (number of slots).

        Args:
            max_batch_size: Maximum number of concurrent slots available.
                Must be positive.

        Raises:
            ValueError: If max_batch_size is not positive.
        """
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {max_batch_size}"
            )
        self._max_batch_size: int = max_batch_size

        # Free list of available slot indices (O(1) allocation via deque)
        self._free_slots: deque[int] = deque(range(max_batch_size))

        # Per-slot generation counter (starts at 1, incremented on release)
        self._slot_generations: list[int] = [1] * max_batch_size

        # Forward mapping: request_id → slot_index
        self._request_to_slot: dict[str, int] = {}

        # Reverse mapping: slot_index → request_id (None if slot is free)
        self._slot_to_request: list[str | None] = [None] * max_batch_size

    def assign_slot(self, request_id: str) -> tuple[int, int]:
        """Assign a slot to a request.

        Picks the next available slot from the free list and maps the
        request ID to that slot index.

        Args:
            request_id: The string request identifier to assign.

        Returns:
            A tuple of (slot_index, slot_generation) for the assigned slot.

        Raises:
            ValueError: If request_id is already assigned a slot.
            RuntimeError: If no free slots are available.
        """
        if request_id in self._request_to_slot:
            raise ValueError(
                f"Request '{request_id}' is already assigned to slot "
                f"{self._request_to_slot[request_id]}"
            )
        if not self._free_slots:
            raise RuntimeError(
                f"No free slots available (capacity: {self._max_batch_size})"
            )

        slot_index = self._free_slots.popleft()
        generation = self._slot_generations[slot_index]

        self._request_to_slot[request_id] = slot_index
        self._slot_to_request[slot_index] = request_id

        return (slot_index, generation)

    def release_slot(self, request_id: str) -> None:
        """Release a slot, incrementing its generation counter.

        The slot becomes available for reuse. The generation counter is
        incremented so that any in-flight messages tagged with the old
        generation will be detected as stale.

        Args:
            request_id: The request whose slot should be released.

        Raises:
            ValueError: If request_id does not have an assigned slot.
        """
        if request_id not in self._request_to_slot:
            raise ValueError(
                f"Request '{request_id}' does not have an assigned slot"
            )

        slot_index = self._request_to_slot[request_id]

        # Remove mappings
        del self._request_to_slot[request_id]
        self._slot_to_request[slot_index] = None

        # Increment generation to invalidate stale messages
        self._slot_generations[slot_index] += 1

        # Return slot to free list
        self._free_slots.append(slot_index)

    def get_slot(self, request_id: str) -> tuple[int, int] | None:
        """Get (slot_index, slot_generation) for a request, or None.

        Args:
            request_id: The request to look up.

        Returns:
            A tuple of (slot_index, slot_generation) if the request has an
            assigned slot, or None if the request is not mapped.
        """
        slot_index = self._request_to_slot.get(request_id)
        if slot_index is None:
            return None
        return (slot_index, self._slot_generations[slot_index])

    def get_request_id(self, slot_index: int) -> str | None:
        """Reverse lookup: get request_id for a slot_index, or None if empty.

        Args:
            slot_index: The slot index to look up.

        Returns:
            The request ID occupying the slot, or None if the slot is free.

        Raises:
            IndexError: If slot_index is out of range.
        """
        if slot_index < 0 or slot_index >= self._max_batch_size:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self._max_batch_size})"
            )
        return self._slot_to_request[slot_index]

    def is_generation_current(self, slot_index: int, generation: int) -> bool:
        """Check if a generation counter matches the current slot generation.

        Used by receivers to detect stale messages from slots that have
        been reassigned to new requests since the message was sent.

        Args:
            slot_index: The slot index to check.
            generation: The generation counter from the message.

        Returns:
            True if the generation matches the current slot generation.

        Raises:
            IndexError: If slot_index is out of range.
        """
        if slot_index < 0 or slot_index >= self._max_batch_size:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self._max_batch_size})"
            )
        return self._slot_generations[slot_index] == generation

    @property
    def active_slot_count(self) -> int:
        """Number of currently occupied slots."""
        return len(self._request_to_slot)

    @property
    def available_slot_count(self) -> int:
        """Number of free slots available for assignment."""
        return len(self._free_slots)


# ---------------------------------------------------------------------------
# PerRequestCacheManager — per-request KV cache and GatedDeltaNet state
# ---------------------------------------------------------------------------


@final
class PerRequestCacheManager:
    """Manages per-request KV caches and GatedDeltaNet persistent state.

    In continuous batching, multiple requests are active simultaneously.
    Each request needs its own isolated cache state. This manager stores
    caches keyed by request_id (not batch position), ensuring that slot
    reuse does not corrupt another request's state.

    The batch position (slot_index) is ephemeral — a request may move
    between slots during its lifetime. The request_id is the stable key.

    Thread safety: NOT thread-safe. Designed for single-threaded per-rank decode.

    **Validates: Requirements 3.2, 3.9, 5.5, 5.7, 5.8**
    """

    def __init__(self, max_requests: int, num_layers: int) -> None:
        """Initialize with capacity for max_requests concurrent caches.

        Args:
            max_requests: Maximum number of concurrent requests that can
                have active caches. Used for capacity validation.
            num_layers: Number of model layers (used for GatedDeltaNet
                state indexing validation).

        Raises:
            ValueError: If max_requests or num_layers is not positive.
        """
        if max_requests <= 0:
            raise ValueError(
                f"max_requests must be positive, got {max_requests}"
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )

        self._max_requests: int = max_requests
        self._num_layers: int = num_layers

        # Primary cache storage: request_id → GatedDeltaNetCache
        self._caches: dict[str, GatedDeltaNetCache] = {}

        # Per-request, per-layer GatedDeltaNet persistent state
        # Key: (request_id, layer_index) → GatedDeltaNetPersistentState
        self._gated_deltanet_states: dict[
            tuple[str, int], GatedDeltaNetPersistentState
        ] = {}

        # Pool of recycled GatedDeltaNet states available for reuse
        # Key: layer_index → list of recycled states
        self._recycled_states: dict[
            int, list[GatedDeltaNetPersistentState]
        ] = {}

    # -------------------------------------------------------------------
    # Public API: Cache lifecycle
    # -------------------------------------------------------------------

    def create_cache(self, request_id: str, dynamic_cache: Any) -> None:
        """Create a new cache entry for a request (called after prefill starts).

        Wraps the provided DynamicCache in a GatedDeltaNetCache and stores
        it keyed by request_id.

        Args:
            request_id: Unique identifier for the request.
            dynamic_cache: A HuggingFace DynamicCache instance to wrap.

        Raises:
            ValueError: If a cache already exists for this request_id.
            RuntimeError: If the manager is at capacity.
        """
        if request_id in self._caches:
            raise ValueError(
                f"Cache already exists for request '{request_id}'"
            )
        if len(self._caches) >= self._max_requests:
            raise RuntimeError(
                f"PerRequestCacheManager at capacity "
                f"({self._max_requests} active caches)"
            )

        # Import here to avoid circular imports at module level
        from exo.worker.engines.pytorch_xpu.gated_deltanet_cache import (
            GatedDeltaNetCache,
        )

        cache = GatedDeltaNetCache(dynamic_cache=dynamic_cache)
        self._caches[request_id] = cache

        logger.debug(
            "Created per-request cache: request=%r, active_count=%d",
            request_id,
            len(self._caches),
        )

    def get_cache(self, request_id: str) -> GatedDeltaNetCache | None:
        """Get the cache for a request, or None if not found.

        Lookup is O(1) by request_id — batch position is not involved.

        Args:
            request_id: The request whose cache to retrieve.

        Returns:
            The GatedDeltaNetCache for the request, or None if no cache
            exists for this request_id.
        """
        return self._caches.get(request_id)

    def remove_cache(self, request_id: str) -> None:
        """Remove and deallocate a request's cache (on completion/cancellation).

        Removes the GatedDeltaNetCache and all associated GatedDeltaNet
        persistent states for the request. States are NOT recycled — they
        are dropped entirely. Use ``recycle_cache`` if you want to preserve
        state tensors for reuse.

        Args:
            request_id: The request whose cache should be removed.

        Does nothing if no cache exists for the request.
        """
        removed_cache = self._caches.pop(request_id, None)
        if removed_cache is None:
            return

        # Remove all GatedDeltaNet states for this request
        keys_to_remove = [
            key for key in self._gated_deltanet_states
            if key[0] == request_id
        ]
        for key in keys_to_remove:
            del self._gated_deltanet_states[key]

        logger.debug(
            "Removed per-request cache: request=%r, "
            "removed_states=%d, active_count=%d",
            request_id,
            len(keys_to_remove),
            len(self._caches),
        )

    def recycle_cache(self, request_id: str) -> None:
        """Recycle a request's GatedDeltaNet states for reuse.

        Zeros all GatedDeltaNet state tensors and moves them to the recycled
        pool, keyed by layer_index. The GatedDeltaNetCache wrapper is removed
        from the active cache map. The next request that needs state for the
        same layer can claim a recycled state instead of allocating new tensors.

        Args:
            request_id: The request whose states should be recycled.

        Does nothing if no cache exists for the request.
        """
        removed_cache = self._caches.pop(request_id, None)
        if removed_cache is None:
            return

        # Recycle all GatedDeltaNet states for this request
        keys_to_recycle = [
            key for key in self._gated_deltanet_states
            if key[0] == request_id
        ]
        recycled_count = 0
        for key in keys_to_recycle:
            state = self._gated_deltanet_states.pop(key)
            state.recycle()
            layer_index = key[1]
            if layer_index not in self._recycled_states:
                self._recycled_states[layer_index] = []
            self._recycled_states[layer_index].append(state)
            recycled_count += 1

        logger.debug(
            "Recycled per-request cache: request=%r, "
            "recycled_states=%d, active_count=%d",
            request_id,
            recycled_count,
            len(self._caches),
        )

    # -------------------------------------------------------------------
    # Public API: GatedDeltaNet persistent state management
    # -------------------------------------------------------------------

    def set_gated_deltanet_state(
        self,
        request_id: str,
        layer_index: int,
        state: GatedDeltaNetPersistentState,
    ) -> None:
        """Store a GatedDeltaNet persistent state for a request and layer.

        Args:
            request_id: The owning request identifier.
            layer_index: The global layer index (0-indexed).
            state: The persistent state container to store.

        Raises:
            ValueError: If no cache exists for the request_id.
            ValueError: If layer_index is out of range.
        """
        if request_id not in self._caches:
            raise ValueError(
                f"No cache exists for request '{request_id}'. "
                f"Call create_cache() first."
            )
        if layer_index < 0 or layer_index >= self._num_layers:
            raise ValueError(
                f"layer_index {layer_index} out of range "
                f"[0, {self._num_layers})"
            )

        self._gated_deltanet_states[(request_id, layer_index)] = state

    def get_gated_deltanet_state(
        self, request_id: str, layer_index: int
    ) -> GatedDeltaNetPersistentState | None:
        """Get persistent state for a specific request and layer.

        Lookup is O(1) by (request_id, layer_index) tuple.

        Args:
            request_id: The request whose state to retrieve.
            layer_index: The global layer index (0-indexed).

        Returns:
            The GatedDeltaNetPersistentState for the request and layer,
            or None if no state has been registered.
        """
        return self._gated_deltanet_states.get((request_id, layer_index))

    def claim_recycled_state(
        self, request_id: str, layer_index: int
    ) -> GatedDeltaNetPersistentState | None:
        """Claim a recycled state for a request and layer, if available.

        Checks the recycled pool for a state matching the layer_index.
        If found, claims it for the new request and registers it in the
        active state map.

        Args:
            request_id: The new owning request identifier.
            layer_index: The layer index to find a recycled state for.

        Returns:
            The claimed GatedDeltaNetPersistentState, or None if no
            recycled state is available for this layer.

        Raises:
            ValueError: If no cache exists for the request_id.
        """
        if request_id not in self._caches:
            raise ValueError(
                f"No cache exists for request '{request_id}'. "
                f"Call create_cache() first."
            )

        pool = self._recycled_states.get(layer_index)
        if not pool:
            return None

        state = pool.pop()
        if not pool:
            del self._recycled_states[layer_index]

        state.claim(request_id)
        self._gated_deltanet_states[(request_id, layer_index)] = state

        logger.debug(
            "Claimed recycled GatedDeltaNet state: request=%r, layer=%d",
            request_id,
            layer_index,
        )
        return state

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def active_cache_count(self) -> int:
        """Number of active per-request caches."""
        return len(self._caches)

    @property
    def total_memory_bytes(self) -> int:
        """Estimated total memory used by all active caches.

        Sums up the fp32 recurrent state tensor memory for all active
        GatedDeltaNet persistent states. Does not include DynamicCache
        memory (which is managed by HuggingFace internals).
        """
        total = 0
        for state in self._gated_deltanet_states.values():
            total += state.shape_metadata.recurrent_state_bytes_fp32
        return total

    @property
    def recycled_state_count(self) -> int:
        """Total number of recycled states available in the pool."""
        return sum(len(pool) for pool in self._recycled_states.values())

    @property
    def active_request_ids(self) -> list[str]:
        """List of request IDs that currently have active caches."""
        return list(self._caches.keys())

    def has_cache(self, request_id: str) -> bool:
        """Check whether a cache exists for the given request.

        Args:
            request_id: The request to check.

        Returns:
            True if a cache is registered for the request.
        """
        return request_id in self._caches
