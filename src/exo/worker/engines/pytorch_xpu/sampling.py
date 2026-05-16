"""
Fast token sampling with routing to specialized implementations.

Provides a routing layer that dispatches sampling to optimized paths based on
the sampling configuration:

- **Greedy**: Uses ``torch.argmax`` without sorting. Selected when temperature
  is near zero or when the configuration explicitly requests greedy decoding.
- **Top-K**: Uses ``torch.topk`` to select candidates without full-vocabulary
  sort. Selected when ``top_k`` is set and ``top_p`` is not active.
- **Fallback**: Full filtering pipeline with temperature scaling, top-k, top-p,
  softmax, and multinomial sampling. Used for unsupported combinations or when
  both top-k and top-p are active.

The public ``sample_token(...)`` function in ``distributed_generator.py`` delegates
to ``route_and_sample_token(...)`` here, preserving its original signature for
backward compatibility.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, final

import torch
from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SamplingRoute — the three dispatch targets
# ---------------------------------------------------------------------------

SamplingRoute = Literal["greedy", "top_k", "fallback"]
"""
Discriminator for which sampling implementation to invoke.

- ``"greedy"``: argmax without sorting
- ``"top_k"``: torch.topk-based candidate selection
- ``"fallback"``: full filtering pipeline (temperature + top-k + top-p + multinomial)
"""


# ---------------------------------------------------------------------------
# SamplingConfiguration — per-request sampling parameters
# ---------------------------------------------------------------------------


@final
class SamplingConfiguration(BaseModel):
    """
    Immutable per-request sampling configuration.

    Determines which sampling route is selected and provides parameters
    for the chosen implementation.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    do_sample: bool = True
    """Whether to use stochastic sampling. When False, greedy decoding is used."""

    temperature: float = 1.0
    """
    Temperature for logit scaling. Values near zero select greedy decoding.
    Must be non-negative.
    """

    top_k: int | None = None
    """
    Number of highest-probability tokens to keep for top-k filtering.
    None disables top-k filtering.
    """

    top_p: float | None = None
    """
    Cumulative probability threshold for nucleus (top-p) filtering.
    None disables top-p filtering. Must be in (0, 1] when set.
    """

    validate_logits: bool = False
    """
    Whether to perform NaN/infinity validation on the full logits tensor
    before sampling. Disabled by default in performance mode.
    """

    @field_validator("temperature")
    @classmethod
    def validate_temperature_non_negative(cls, value: float) -> float:
        """Temperature must be non-negative."""
        if value < 0.0:
            raise ValueError(f"temperature must be non-negative, got {value}")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k_positive_or_none(cls, value: int | None) -> int | None:
        """top_k must be positive when set."""
        if value is not None and value <= 0:
            raise ValueError(f"top_k must be positive when set, got {value}")
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p_range(cls, value: float | None) -> float | None:
        """top_p must be in (0, 1] when set."""
        if value is not None and (value <= 0.0 or value > 1.0):
            raise ValueError(f"top_p must be in (0, 1] when set, got {value}")
        return value


# ---------------------------------------------------------------------------
# Route determination
# ---------------------------------------------------------------------------

_GREEDY_TEMPERATURE_THRESHOLD: float = 1e-7
"""Temperature values at or below this threshold trigger greedy decoding."""


def determine_sampling_route(
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    do_sample: bool = True,
) -> SamplingRoute:
    """
    Determine which sampling implementation to use based on parameters.

    Routing rules:
    1. If ``do_sample`` is False or temperature <= threshold → greedy
    2. If ``top_k`` is set and ``top_p`` is None → top_k
    3. Otherwise → fallback (handles top_p, top_k + top_p, or plain multinomial)

    Args:
        temperature: Sampling temperature.
        top_k: Top-k parameter (None = disabled).
        top_p: Top-p parameter (None = disabled).
        do_sample: Whether stochastic sampling is enabled.

    Returns:
        The sampling route to dispatch to.
    """
    if not do_sample or temperature <= _GREEDY_TEMPERATURE_THRESHOLD:
        return "greedy"

    if top_k is not None and top_k > 0 and top_p is None:
        return "top_k"

    return "fallback"


# ---------------------------------------------------------------------------
# Specialized sampling implementations
# ---------------------------------------------------------------------------


def sample_token_greedy(
    logits: torch.Tensor,
    performance_recorder: PerformanceRecorder | None = None,
) -> int:
    """
    Greedy sampling: return the token with the highest logit value.

    Uses ``torch.argmax`` without calling ``torch.sort``.

    Args:
        logits: 1D tensor of shape ``(vocabulary_size,)`` — raw logit values.
        performance_recorder: Optional recorder for timing spans.

    Returns:
        Token identifier with the maximum logit value.
    """
    if performance_recorder is not None:
        context = performance_recorder.span(
            "sample_token_greedy",
            mode="decode",
            metadata={"route": "greedy"},
        )
    else:
        context = nullcontext()

    with context:
        token_identifier: int = int(logits.argmax(dim=-1).item())

    return token_identifier


def sample_token_top_k(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    performance_recorder: PerformanceRecorder | None = None,
) -> int:
    """
    Top-k sampling: select from the top-k highest logit tokens.

    Uses ``torch.topk`` to extract candidates without sorting the full vocabulary.
    Applies temperature scaling to the top-k logits, converts to probabilities
    via softmax, and samples using ``torch.multinomial``.

    Args:
        logits: 1D tensor of shape ``(vocabulary_size,)`` — raw logit values.
        top_k: Number of top candidates to consider.
        temperature: Temperature for logit scaling before softmax.
        performance_recorder: Optional recorder for timing spans.

    Returns:
        Sampled token identifier from the top-k candidates.
    """
    if performance_recorder is not None:
        context = performance_recorder.span(
            "sample_token_top_k",
            mode="decode",
            metadata={"route": "top_k", "top_k": top_k, "temperature": temperature},
        )
    else:
        context = nullcontext()

    with context:
        vocabulary_size = logits.size(0)
        effective_k = min(top_k, vocabulary_size)

        # Extract top-k values and their indices without full sort
        top_k_values, top_k_indices = torch.topk(logits, effective_k)

        # Apply temperature scaling
        if temperature != 1.0:
            top_k_values = top_k_values / temperature

        # Convert to probabilities
        top_k_probabilities = torch.softmax(top_k_values, dim=-1)

        # Sample from the top-k distribution
        try:
            sampled_index = int(
                torch.multinomial(top_k_probabilities.unsqueeze(0), num_samples=1)
                .squeeze()
                .item()
            )
            token_identifier: int = int(top_k_indices[sampled_index].item())
        except Exception:
            # Fallback to argmax within top-k on multinomial failure
            token_identifier = int(top_k_indices[0].item())

    return token_identifier


def sample_token_fallback(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    performance_recorder: PerformanceRecorder | None = None,
) -> int:
    """
    Fallback sampling: full filtering pipeline.

    Applies temperature scaling, top-k filtering, top-p (nucleus) filtering,
    softmax, and multinomial sampling. Used for unsupported combinations
    (top-p with or without top-k) or when explicit fallback is requested.

    This preserves the original ``sample_token`` behavior for configurations
    that do not match the greedy or top-k-only fast paths.

    Args:
        logits: 1D tensor of shape ``(vocabulary_size,)`` — raw logit values.
        temperature: Temperature for logit scaling.
        top_k: Top-k parameter (None = disabled).
        top_p: Top-p parameter (None = disabled).
        performance_recorder: Optional recorder for timing spans.

    Returns:
        Sampled token identifier.
    """
    if performance_recorder is not None:
        context = performance_recorder.span(
            "sample_token_fallback",
            mode="decode",
            metadata={
                "route": "fallback",
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        )
    else:
        context = nullcontext()

    with context:
        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Check for NaN or inf after temperature scaling
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(
                "Detected NaN or inf after temperature scaling in fallback path, "
                "using argmax"
            )
            return int(logits.nanargmax().item())

        # Top-k filtering: set values below k-th largest to -inf
        if top_k is not None and top_k > 0:
            if top_k < logits.size(0):
                top_k_values, _ = torch.topk(logits, top_k)
                threshold = top_k_values[-1]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

            # Mask tokens where cumulative probability exceeds top_p
            sorted_mask = torch.zeros_like(
                cumulative_probabilities, dtype=torch.bool
            )
            sorted_mask[1:] = cumulative_probabilities[:-1] >= top_p

            # Set masked logits to -inf in sorted order
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

            # Scatter back to original positions
            logits = torch.zeros_like(logits).scatter(
                0, sorted_indices, sorted_logits
            )

        # Softmax and multinomial sampling
        logits = torch.clamp(logits, min=-1e9, max=1e9)
        probabilities = torch.softmax(logits, dim=-1)

        # Validate probabilities
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            logger.warning(
                "Invalid probabilities in fallback path, using argmax"
            )
            return int(logits.argmax().item())

        # Normalize
        probability_sum = probabilities.sum()
        if probability_sum == 0:
            logger.warning(
                "All probabilities are zero in fallback path, using argmax"
            )
            return int(logits.argmax().item())
        probabilities = probabilities / probability_sum

        try:
            token_identifier: int = int(
                torch.multinomial(probabilities.unsqueeze(0), num_samples=1)
                .squeeze()
                .item()
            )
        except Exception as sampling_error:
            logger.warning(
                f"Multinomial sampling failed in fallback path: {sampling_error}, "
                f"using argmax"
            )
            token_identifier = int(logits.argmax().item())

    return token_identifier


# ---------------------------------------------------------------------------
# Routing dispatcher
# ---------------------------------------------------------------------------


def route_and_sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    validate_logits: bool = True,
    performance_recorder: PerformanceRecorder | None = None,
) -> tuple[int, SamplingRoute]:
    """
    Route to the appropriate sampling implementation and return the sampled token.

    This is the internal dispatch function called by the public ``sample_token()``
    in ``distributed_generator.py``. It determines the sampling route, performs
    optional logit validation, and dispatches to the specialized implementation.

    Args:
        logits: 1D tensor of shape ``(vocabulary_size,)`` — raw logit values
            after last-position extraction.
        temperature: Sampling temperature.
        top_k: Top-k parameter (None = disabled).
        top_p: Top-p parameter (None = disabled).
        validate_logits: Whether to check for NaN/inf in logits before sampling.
            When disabled, skips validation for performance.
        performance_recorder: Optional recorder for timing and counting.

    Returns:
        A tuple of (sampled_token_identifier, route_used).
    """
    # --- Logit validation (controlled by flag) ---
    if validate_logits:
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(
                "Detected NaN or inf in logits, falling back to argmax on finite values"
            )
            finite_logits = logits.clone()
            finite_logits[
                torch.isnan(finite_logits) | torch.isinf(finite_logits)
            ] = float("-inf")
            if (finite_logits == float("-inf")).all():
                logger.error(
                    "All logits are NaN or Inf — no valid token to sample"
                )
                return 0, "greedy"
            return int(finite_logits.argmax().item()), "greedy"

        # Check for all-equal logits (indicates all-reduce failure or stale state)
        logits_range = logits.max() - logits.min()
        if logits_range < 1e-6:
            logger.warning(
                f"Detected near-constant logits (range={float(logits_range):.6e}), "
                f"falling back to argmax"
            )
            return int(logits.argmax().item()), "greedy"

    # --- Determine route ---
    route: SamplingRoute = determine_sampling_route(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # --- Increment instrumentation counter ---
    if performance_recorder is not None:
        performance_recorder.increment_counter(f"sampling_route_{route}")

    # --- Dispatch to specialized implementation ---
    if route == "greedy":
        token_identifier = sample_token_greedy(
            logits=logits,
            performance_recorder=performance_recorder,
        )
    elif route == "top_k":
        assert top_k is not None  # guaranteed by routing logic
        token_identifier = sample_token_top_k(
            logits=logits,
            top_k=top_k,
            temperature=temperature,
            performance_recorder=performance_recorder,
        )
    else:
        token_identifier = sample_token_fallback(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            performance_recorder=performance_recorder,
        )

    return token_identifier, route



# ---------------------------------------------------------------------------
# Batched per-request sampling
# ---------------------------------------------------------------------------


def sample_tokens_batched(
    logits: torch.Tensor,
    configurations: Sequence[SamplingConfiguration],
    performance_recorder: PerformanceRecorder | None = None,
) -> torch.Tensor:
    """
    Sample one token per batch element with per-request sampling configurations.

    Accepts a 2D logits tensor shaped ``[batch_size, vocabulary_size]`` and a
    sequence of ``SamplingConfiguration`` objects (one per batch element). Each
    batch element is independently routed to the appropriate sampling
    implementation based on its configuration.

    This supports mixed configurations in the same batch — for example, some
    requests using greedy decoding while others use top-k sampling.

    Args:
        logits: 2D tensor of shape ``(batch_size, vocabulary_size)`` — raw logit
            values for each request in the batch.
        configurations: Sequence of ``SamplingConfiguration`` objects, one per
            batch element. Length must equal ``logits.shape[0]``.
        performance_recorder: Optional recorder for timing spans and counters.

    Returns:
        1D tensor of shape ``(batch_size,)`` containing the sampled token
        identifier for each batch element.

    Raises:
        ValueError: If ``logits`` is not 2D or if the number of configurations
            does not match the batch size.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2D with shape [batch_size, vocabulary_size], "
            f"got shape {list(logits.shape)}"
        )

    batch_size = logits.shape[0]
    vocabulary_size = logits.shape[1]

    if len(configurations) != batch_size:
        raise ValueError(
            f"Number of configurations ({len(configurations)}) must match "
            f"batch size ({batch_size})"
        )

    if performance_recorder is not None:
        context = performance_recorder.span(
            "sample_tokens_batched",
            mode="decode",
            metadata={
                "batch_size": batch_size,
                "vocabulary_size": vocabulary_size,
            },
        )
    else:
        context = nullcontext()

    with context:
        token_identifiers = torch.empty(
            batch_size, dtype=torch.long, device=logits.device
        )

        for batch_index in range(batch_size):
            element_logits = logits[batch_index]
            configuration = configurations[batch_index]

            # Determine route from this element's configuration
            route: SamplingRoute = determine_sampling_route(
                temperature=configuration.temperature,
                top_k=configuration.top_k,
                top_p=configuration.top_p,
                do_sample=configuration.do_sample,
            )

            # Increment per-route counter
            if performance_recorder is not None:
                performance_recorder.increment_counter(
                    f"batched_sampling_route_{route}"
                )

            # Optional logit validation per element
            if configuration.validate_logits:
                if torch.isnan(element_logits).any() or torch.isinf(
                    element_logits
                ).any():
                    logger.warning(
                        f"Detected NaN or inf in logits for batch element "
                        f"{batch_index}, falling back to argmax on finite values"
                    )
                    finite_logits = element_logits.clone()
                    finite_logits[
                        torch.isnan(finite_logits) | torch.isinf(finite_logits)
                    ] = float("-inf")
                    if (finite_logits == float("-inf")).all():
                        logger.error(
                            f"All logits are NaN or Inf for batch element "
                            f"{batch_index} — defaulting to token 0"
                        )
                        token_identifiers[batch_index] = 0
                        continue
                    token_identifiers[batch_index] = int(
                        finite_logits.argmax().item()
                    )
                    continue

                # Check for near-constant logits
                logits_range = element_logits.max() - element_logits.min()
                if logits_range < 1e-6:
                    logger.warning(
                        f"Detected near-constant logits for batch element "
                        f"{batch_index} (range={float(logits_range):.6e}), "
                        f"falling back to argmax"
                    )
                    token_identifiers[batch_index] = int(
                        element_logits.argmax().item()
                    )
                    continue

            # Dispatch to specialized implementation
            if route == "greedy":
                token_identifier = sample_token_greedy(
                    logits=element_logits,
                    performance_recorder=None,  # avoid double-counting spans
                )
            elif route == "top_k":
                assert configuration.top_k is not None
                token_identifier = sample_token_top_k(
                    logits=element_logits,
                    top_k=configuration.top_k,
                    temperature=configuration.temperature,
                    performance_recorder=None,
                )
            else:
                token_identifier = sample_token_fallback(
                    logits=element_logits,
                    temperature=configuration.temperature,
                    top_k=configuration.top_k,
                    top_p=configuration.top_p,
                    performance_recorder=None,
                )

            token_identifiers[batch_index] = token_identifier

    return token_identifiers
