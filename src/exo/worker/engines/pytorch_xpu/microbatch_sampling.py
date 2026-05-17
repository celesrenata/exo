"""
Per-request sampling for decode microbatches.

Integrates the Task 3 sampling infrastructure with the continuous batching
engine to support different sampling configurations within the same microbatch.
Each active slot in a decode microbatch can have its own sampling parameters
(greedy, top-k, temperature, etc.).

Requirements: 3.9, 4.1, 4.7
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exo.worker.engines.pytorch_xpu.continuous_batching import (
    DecodeMicrobatch,
    TokenResultBatch,
)

if TYPE_CHECKING:
    import torch

    from exo.worker.engines.pytorch_xpu.continuous_batching_engine import (
        ContinuousBatchingEngine,
    )

logger = logging.getLogger(__name__)


def sample_microbatch(
    *,
    logits_batch: torch.Tensor,
    microbatch: DecodeMicrobatch,
    engine: ContinuousBatchingEngine,
) -> TokenResultBatch:
    """Sample tokens from logits using per-request sampling configurations.

    Each request in the microbatch may have a different sampling config
    (greedy, top-k, temperature, etc.). This function applies the correct
    sampler per-request and returns a TokenResultBatch.

    Args:
        logits_batch: Logits tensor with one row per active slot, shaped
            ``[active_slot_count, vocabulary_size]``.
        microbatch: The decode microbatch describing active slots.
        engine: The engine (for looking up per-request sampling configs).

    Returns:
        TokenResultBatch with sampled tokens for each active slot.

    Raises:
        ValueError: If logits_batch row count does not match the number of
            active slots in the microbatch.
    """
    from exo.worker.engines.pytorch_xpu.sampling import (
        SamplingConfiguration,
        determine_sampling_route,
        sample_token_fallback,
        sample_token_greedy,
        sample_token_top_k,
    )

    active_slots = [
        slot for slot in microbatch.slot_states if slot.is_active
    ]
    active_count = len(active_slots)

    if logits_batch.shape[0] != active_count:
        raise ValueError(
            f"logits_batch has {logits_batch.shape[0]} rows but microbatch "
            f"has {active_count} active slots"
        )

    # Handle empty microbatch
    if active_count == 0:
        return TokenResultBatch(
            token_ids=(),
            slot_indices=(),
            slot_generations=(),
        )

    token_ids: list[int] = []
    slot_indices: list[int] = []
    slot_generations: list[int] = []

    for batch_index, slot_state in enumerate(active_slots):
        request_id = slot_state.request_id

        # Look up sampling configuration from the engine
        sampling_config: SamplingConfiguration | None = None
        if request_id is not None:
            # Access the engine's internal request data for sampling config
            pending = engine._requests.get(request_id)
            if pending is not None:
                sampling_config = pending.sampling_config

        # Fall back to greedy if no config found
        if sampling_config is None:
            sampling_config = SamplingConfiguration(do_sample=False)
            logger.warning(
                "No sampling config found for slot %d (request=%r), "
                "defaulting to greedy",
                slot_state.slot_index,
                request_id,
            )

        # Extract logits for this slot
        element_logits = logits_batch[batch_index]

        # Determine sampling route
        route = determine_sampling_route(
            temperature=sampling_config.temperature,
            top_k=sampling_config.top_k,
            top_p=sampling_config.top_p,
            do_sample=sampling_config.do_sample,
        )

        # Dispatch to the appropriate sampler
        if route == "greedy":
            token_id = sample_token_greedy(logits=element_logits)
        elif route == "top_k":
            assert sampling_config.top_k is not None
            token_id = sample_token_top_k(
                logits=element_logits,
                top_k=sampling_config.top_k,
                temperature=sampling_config.temperature,
            )
        else:
            token_id = sample_token_fallback(
                logits=element_logits,
                temperature=sampling_config.temperature,
                top_k=sampling_config.top_k,
                top_p=sampling_config.top_p,
            )

        token_ids.append(token_id)
        slot_indices.append(slot_state.slot_index)
        slot_generations.append(slot_state.slot_generation)

    return TokenResultBatch(
        token_ids=tuple(token_ids),
        slot_indices=tuple(slot_indices),
        slot_generations=tuple(slot_generations),
    )
