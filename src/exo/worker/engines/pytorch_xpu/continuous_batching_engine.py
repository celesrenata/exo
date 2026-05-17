"""
Continuous batching engine — high-level coordination layer.

Ties together the ContinuousBatchScheduler, PerRequestCacheManager, and
RequestIdentifierMap to process multiple concurrent generation requests
through the pipeline.

This class manages the generation loop:
1. Admit new requests (can happen while decode is active)
2. Prefill one request at a time (sequential)
3. Batch all decode-ready requests into microbatches
4. Return token streams per request via callbacks
5. Support cancellation at any point

This module does NOT run inference itself. The pipeline generator calls
``get_prefill_request()``, ``get_decode_microbatch()``, and
``report_token_results()`` to drive the actual computation.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.9, 3.10, 3.11
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from exo.worker.engines.pytorch_xpu.continuous_batching import (
    BatchSlotState,
    ContinuousBatchScheduler,
    DecodeMicrobatch,
    PerRequestCacheManager,
    RequestIdentifierMap,
    TokenResultBatch,
)

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.sampling import SamplingConfiguration

logger_module = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _PendingRequest — internal bookkeeping for a submitted request
# ---------------------------------------------------------------------------


@final
@dataclass
class _PendingRequest:
    """Internal bookkeeping for a request submitted to the engine."""

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    sampling_config: SamplingConfiguration
    on_token: Callable[[str, int], None] | None
    tokens_generated: int = 0
    last_token_id: int = 0
    is_finished: bool = False


# ---------------------------------------------------------------------------
# ContinuousBatchingEngine — main coordination class
# ---------------------------------------------------------------------------


@final
class ContinuousBatchingEngine:
    """Engine interface for continuous-batching generation mode.

    Coordinates the ContinuousBatchScheduler, PerRequestCacheManager,
    and RequestIdentifierMap to process multiple concurrent generation
    requests through the pipeline.

    This class manages the generation loop:
    1. Admit new requests (can happen while decode is active)
    2. Prefill one request at a time (sequential)
    3. Batch all decode-ready requests into microbatches
    4. Return token streams per request via callbacks
    5. Support cancellation at any point
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        num_layers: int,
        world_size: int,
    ) -> None:
        """Initialize the continuous batching engine.

        Args:
            max_batch_size: Maximum number of concurrent requests in a
                decode microbatch.
            num_layers: Number of model layers (for cache management).
            world_size: Number of pipeline-parallel ranks.

        Raises:
            ValueError: If max_batch_size, num_layers, or world_size is
                not positive.
        """
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {max_batch_size}"
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if world_size <= 0:
            raise ValueError(
                f"world_size must be positive, got {world_size}"
            )

        self._max_batch_size: int = max_batch_size
        self._num_layers: int = num_layers
        self._world_size: int = world_size

        # Core components
        self._scheduler = ContinuousBatchScheduler()
        self._cache_manager = PerRequestCacheManager(
            max_requests=max_batch_size,
            num_layers=num_layers,
        )
        self._identifier_map = RequestIdentifierMap(
            max_batch_size=max_batch_size,
        )

        # Per-request metadata (keyed by request_id)
        self._requests: dict[str, _PendingRequest] = {}

        # Track which request is currently being prefilled
        self._active_prefill: str | None = None

        # Completed request IDs (for reporting)
        self._completed_requests: set[str] = set()

    # -------------------------------------------------------------------
    # Public API: Request submission
    # -------------------------------------------------------------------

    def submit_request(
        self,
        request_id: str,
        prompt_tokens: list[int],
        max_tokens: int,
        sampling_config: SamplingConfiguration,
        on_token: Callable[[str, int], None] | None = None,
    ) -> None:
        """Submit a new generation request.

        Can be called while decode is active — the request is queued
        for prefill and will join the decode batch after prefill completes.

        Args:
            request_id: Globally unique identifier for this request.
            prompt_tokens: Token IDs comprising the prompt.
            max_tokens: Maximum number of tokens to generate.
            sampling_config: Per-request sampling parameters.
            on_token: Optional callback invoked with (request_id, token_id)
                each time a token is generated for this request.

        Raises:
            ValueError: If request_id is already known to the engine.
            ValueError: If max_tokens is not positive.
        """
        if max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {max_tokens}"
            )
        if request_id in self._requests:
            raise ValueError(
                f"Request '{request_id}' is already submitted to the engine"
            )
        if request_id in self._completed_requests:
            raise ValueError(
                f"Request '{request_id}' has already completed"
            )

        # Create internal bookkeeping
        pending = _PendingRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            sampling_config=sampling_config,
            on_token=on_token,
        )
        self._requests[request_id] = pending

        # Admit to scheduler
        self._scheduler.admit_request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_config=sampling_config,
        )

        logger_module.debug(
            "Submitted request: request=%r, prompt_length=%d, max_tokens=%d",
            request_id,
            len(prompt_tokens),
            max_tokens,
        )

    # -------------------------------------------------------------------
    # Public API: Cancellation
    # -------------------------------------------------------------------

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request. Safe to call from any state.

        Removes the request from all queues, releases its slot (if assigned),
        and removes its cache. Does nothing if the request is not active.

        Args:
            request_id: The request to cancel.
        """
        if request_id not in self._requests:
            # Already completed or never submitted — no-op
            return

        # Cancel in scheduler (handles removal from admission/prefill/decode)
        try:
            self._scheduler.cancel_request(request_id)
        except ValueError:
            # Request was already removed from scheduler
            pass

        # Release slot if assigned
        slot_info = self._identifier_map.get_slot(request_id)
        if slot_info is not None:
            self._identifier_map.release_slot(request_id)

        # Remove cache if present
        self._cache_manager.remove_cache(request_id)

        # Clear active prefill if this was the prefilling request
        if self._active_prefill == request_id:
            self._active_prefill = None

        # Remove from internal tracking
        del self._requests[request_id]

        logger_module.debug("Cancelled request: request=%r", request_id)

    # -------------------------------------------------------------------
    # Public API: Step-based generation loop
    # -------------------------------------------------------------------

    def step(self) -> list[tuple[str, int]]:
        """Execute one scheduling step.

        Returns list of (request_id, token_id) pairs for requests
        that generated a token in this step. Empty list if no decode
        batch is ready.

        The step handles:
        1. Check if a prefill can start (admission queue non-empty,
           no active prefill)
        2. Return the current decode batch composition (request IDs
           that are decode-ready)

        Note: This method does NOT perform inference. It returns the
        batch composition for the pipeline generator to execute.
        """
        # Try to start a prefill if none is active
        if self._active_prefill is None:
            next_prefill = self._scheduler.get_next_prefill()
            if next_prefill is not None:
                self._active_prefill = next_prefill

        # Return empty — actual token generation happens via
        # get_decode_microbatch() and report_token_results()
        return []

    # -------------------------------------------------------------------
    # Public API: Prefill management
    # -------------------------------------------------------------------

    def get_prefill_request(self) -> tuple[str, list[int]] | None:
        """Get the next request to prefill, or None if nothing to prefill.

        If no prefill is currently active, pops the next request from the
        admission queue and returns its ID and prompt tokens.

        Returns:
            A tuple of (request_id, prompt_tokens) for the request to
            prefill, or None if no prefill is needed.
        """
        # If we already have an active prefill, return it
        if self._active_prefill is not None:
            pending = self._requests.get(self._active_prefill)
            if pending is not None:
                return (self._active_prefill, pending.prompt_tokens)
            # Active prefill was cancelled — clear it
            self._active_prefill = None

        # Try to get next from scheduler
        next_prefill = self._scheduler.get_next_prefill()
        if next_prefill is None:
            return None

        self._active_prefill = next_prefill
        pending = self._requests.get(next_prefill)
        if pending is None:
            # Request was cancelled between admission and prefill
            self._active_prefill = None
            return None

        return (next_prefill, pending.prompt_tokens)

    def mark_prefill_done(self, request_id: str) -> None:
        """Mark a request's prefill as complete, moving it to decode-ready.

        Assigns a slot in the identifier map and transitions the request
        from prefilling to decoding state in the scheduler.

        Args:
            request_id: The request that has completed prefill.

        Raises:
            ValueError: If the request is not currently prefilling.
        """
        if self._active_prefill != request_id:
            raise ValueError(
                f"Request '{request_id}' is not the active prefill "
                f"(active: {self._active_prefill})"
            )

        # Assign a slot for decode
        self._identifier_map.assign_slot(request_id)

        # Move to decode-ready in scheduler
        self._scheduler.mark_prefill_complete(request_id)

        # Clear active prefill
        self._active_prefill = None

        logger_module.debug(
            "Prefill done: request=%r, moved to decode-ready",
            request_id,
        )

    # -------------------------------------------------------------------
    # Public API: Decode batch formation
    # -------------------------------------------------------------------

    def get_decode_microbatch(self) -> DecodeMicrobatch | None:
        """Build a DecodeMicrobatch from all decode-ready requests.

        Returns None if no requests are decode-ready.

        Returns:
            A DecodeMicrobatch describing the batch composition for the
            next decode step, or None if no decode work is available.
        """
        decode_request_ids = self._scheduler.get_decode_batch()
        if not decode_request_ids:
            return None

        # Build slot states and input tokens
        slot_states: list[BatchSlotState] = []
        input_token_ids: list[int] = []

        for rid in decode_request_ids:
            slot_info = self._identifier_map.get_slot(rid)
            if slot_info is None:
                # Request has no slot — skip (should not happen in normal flow)
                logger_module.warning(
                    "Decode-ready request %r has no assigned slot", rid
                )
                continue

            slot_index, slot_generation = slot_info
            pending = self._requests.get(rid)
            if pending is None:
                continue

            slot_states.append(
                BatchSlotState(
                    slot_index=slot_index,
                    request_id=rid,
                    slot_generation=slot_generation,
                    is_active=True,
                )
            )
            input_token_ids.append(pending.last_token_id)

        if not slot_states:
            return None

        return DecodeMicrobatch(
            active_slot_count=len(slot_states),
            max_batch_size=self._max_batch_size,
            slot_states=tuple(slot_states),
            input_token_ids=tuple(input_token_ids),
        )

    # -------------------------------------------------------------------
    # Public API: Token result reporting
    # -------------------------------------------------------------------

    def report_token_results(self, results: TokenResultBatch) -> list[str]:
        """Report generated tokens. Returns list of completed request IDs.

        Updates internal state for each request that received a token.
        Invokes per-request callbacks. Detects completion (max_tokens reached)
        and returns the IDs of requests that finished.

        Args:
            results: Token results from a decode step.

        Returns:
            List of request IDs that completed in this step (reached
            max_tokens or generated EOS).
        """
        completed: list[str] = []

        for idx in range(len(results.token_ids)):
            token_id = results.token_ids[idx]
            slot_index = results.slot_indices[idx]
            slot_generation = results.slot_generations[idx]

            # Validate generation is current (not stale)
            if not self._identifier_map.is_generation_current(
                slot_index, slot_generation
            ):
                logger_module.debug(
                    "Discarding stale token result: slot=%d, generation=%d",
                    slot_index,
                    slot_generation,
                )
                continue

            # Look up request by slot
            request_id = self._identifier_map.get_request_id(slot_index)
            if request_id is None:
                continue

            pending = self._requests.get(request_id)
            if pending is None:
                continue

            # Update request state
            pending.tokens_generated += 1
            pending.last_token_id = token_id

            # Invoke callback
            if pending.on_token is not None:
                try:
                    pending.on_token(request_id, token_id)
                except Exception as callback_error:
                    logger_module.warning(
                        "Token callback failed for request %r: %s",
                        request_id,
                        callback_error,
                    )

            # Check completion
            if pending.tokens_generated >= pending.max_tokens:
                pending.is_finished = True
                completed.append(request_id)

        # Finalize completed requests
        for request_id in completed:
            self._finalize_request(request_id)

        return completed

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def has_work(self) -> bool:
        """Whether there are requests to process (prefill or decode).

        Returns True if there are any active requests in the engine,
        including those waiting for admission, prefilling, or decoding.
        """
        return len(self._requests) > 0

    @property
    def active_request_count(self) -> int:
        """Total number of active requests (admitted + prefilling + decoding)."""
        return len(self._requests)

    @property
    def decode_batch_size(self) -> int:
        """Number of requests currently in the decode-ready state."""
        return self._scheduler.decode_ready_count

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _finalize_request(self, request_id: str) -> None:
        """Move a request to completed state and release all resources."""
        # Mark completed in scheduler
        try:
            self._scheduler.mark_completed(request_id)
        except ValueError:
            pass

        # Release slot
        slot_info = self._identifier_map.get_slot(request_id)
        if slot_info is not None:
            self._identifier_map.release_slot(request_id)

        # Remove cache
        self._cache_manager.remove_cache(request_id)

        # Remove from internal tracking
        if request_id in self._requests:
            del self._requests[request_id]

        # Track as completed
        self._completed_requests.add(request_id)

        logger_module.debug(
            "Finalized request: request=%r", request_id
        )
