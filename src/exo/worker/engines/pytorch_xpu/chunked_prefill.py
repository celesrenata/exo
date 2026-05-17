"""Chunk-level transform representation for GatedDeltaNet chunked prefill.

Defines internal types for representing the composed affine transform of a chunk
of tokens in the GatedDeltaNet recurrence. The recurrence has the form:

    S_t = A_t * S_{t-1} + B_t

where:
    A_t = exp(g_t) * (I - β_t * k_t * k_t^T)  — rank-1 perturbation of scaled identity
    B_t = β_t * k_t ⊗ v_t                      — rank-1 outer product

For efficient chunk-level composition, we use the WY representation that avoids
materializing the full d_k × d_k matrix A. Instead, we store:
    - The cumulative log-decay (scalar per head)
    - The accumulated rank-1 corrections (W and Y matrices from the WY decomposition)
    - The composed additive term B

All tensors are stored in fp32 for numerical stability during composition.

**Validates: Requirements 7.1, 7.5, 7.6, 7.7**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

# ---------------------------------------------------------------------------
# ChunkTransform — composed affine transform for a chunk of tokens
# ---------------------------------------------------------------------------


@final
@dataclass
class ChunkTransform:
    """Composed affine transform for a chunk of tokens.

    Represents the composition of per-token transforms within a chunk:
        S_final = A_composed * S_initial + B_composed

    Uses the WY representation for A_composed to avoid materializing
    the full d_k × d_k matrix. The composed transform is stored as:
        - cumulative_log_decay: total log-decay across the chunk (scalar per head)
        - correction_keys: accumulated key vectors for rank-1 corrections (C × d_k)
        - correction_weights: accumulated weights for corrections (C × 1)
        - additive_term: B_composed (d_k × d_v) — the accumulated additive state

    All tensors are fp32 for numerical stability.

    The WY representation encodes A_composed implicitly as:
        A_composed = exp(cumulative_log_decay) * (I - correction_keys^T @ diag(correction_weights) @ correction_keys)

    This avoids materializing the full (d_k × d_k) matrix while supporting
    efficient application via:
        A_composed @ x = exp(decay) * (x - correction_keys^T @ (correction_weights * (correction_keys @ x)))
    """

    cumulative_log_decay: torch.Tensor
    """Total log-decay across the chunk, shape (B, H). Sum of per-token log-gates."""

    correction_keys: torch.Tensor
    """Accumulated key vectors for WY rank-1 corrections, shape (B, H, C, d_k).

    Each row is a key vector from a token in the chunk, used to represent
    the rank-1 corrections to the identity in the WY decomposition.
    """

    correction_weights: torch.Tensor
    """Accumulated weights for WY corrections, shape (B, H, C).

    Each weight combines the token's β value and relative decay factor
    within the chunk.
    """

    additive_term: torch.Tensor
    """Composed B term (accumulated additive state), shape (B, H, d_k, d_v).

    The sum of all per-token B_t contributions, each decayed by the
    cumulative gate from its position to the end of the chunk.
    """

    chunk_size: int
    """Number of tokens in this chunk (may be less than max for the final chunk)."""

    num_heads: int
    """Number of attention heads (H)."""

    key_dim: int
    """Key vector dimension (d_k)."""

    value_dim: int
    """Value vector dimension (d_v)."""


# ---------------------------------------------------------------------------
# ChunkOutput — intra-chunk output activations
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class ChunkOutput:
    """Intra-chunk output activations before inter-chunk correction.

    Stores the per-token output activations computed within a single chunk
    using only intra-chunk information. These activations are later corrected
    by the inter-chunk state contribution during the output materialization phase.
    """

    activations: torch.Tensor
    """Per-token output activations, shape (B, C, H, d_v).

    These are the raw intra-chunk outputs before adding the contribution
    from the incoming state (which is unknown during parallel chunk processing).
    """

    chunk_index: int
    """Zero-based index of this chunk within the sequence."""

    chunk_size: int
    """Number of valid tokens in this chunk (may be less than max for final chunk)."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ChunkTransformValidationError(ValueError):
    """Raised when a ChunkTransform has inconsistent shapes or wrong dtype."""


def validate_chunk_transform_shapes(transform: ChunkTransform) -> None:
    """Validate that all tensors in a ChunkTransform have consistent shapes and fp32 dtype.

    Checks:
        1. All tensors are fp32
        2. cumulative_log_decay has shape (B, H) matching num_heads
        3. correction_keys has shape (B, H, C, d_k) matching key_dim and chunk_size
        4. correction_weights has shape (B, H, C) matching chunk_size
        5. additive_term has shape (B, H, d_k, d_v) matching key_dim and value_dim
        6. Batch dimensions are consistent across all tensors

    Args:
        transform: The ChunkTransform to validate.

    Raises:
        ChunkTransformValidationError: If any validation check fails.
    """
    # --- dtype checks ---
    if transform.cumulative_log_decay.dtype != torch.float32:
        raise ChunkTransformValidationError(
            f"cumulative_log_decay must be fp32, got {transform.cumulative_log_decay.dtype}"
        )
    if transform.correction_keys.dtype != torch.float32:
        raise ChunkTransformValidationError(
            f"correction_keys must be fp32, got {transform.correction_keys.dtype}"
        )
    if transform.correction_weights.dtype != torch.float32:
        raise ChunkTransformValidationError(
            f"correction_weights must be fp32, got {transform.correction_weights.dtype}"
        )
    if transform.additive_term.dtype != torch.float32:
        raise ChunkTransformValidationError(
            f"additive_term must be fp32, got {transform.additive_term.dtype}"
        )

    # --- shape checks ---
    decay_shape = transform.cumulative_log_decay.shape
    if len(decay_shape) != 2:
        raise ChunkTransformValidationError(
            f"cumulative_log_decay must be 2D (B, H), got shape {decay_shape}"
        )

    batch_size = decay_shape[0]
    num_heads_from_decay = decay_shape[1]

    if num_heads_from_decay != transform.num_heads:
        raise ChunkTransformValidationError(
            f"cumulative_log_decay H dimension is {num_heads_from_decay}, "
            f"expected num_heads={transform.num_heads}"
        )

    # correction_keys: (B, H, C, d_k)
    keys_shape = transform.correction_keys.shape
    if len(keys_shape) != 4:
        raise ChunkTransformValidationError(
            f"correction_keys must be 4D (B, H, C, d_k), got shape {keys_shape}"
        )
    if keys_shape[0] != batch_size:
        raise ChunkTransformValidationError(
            f"correction_keys batch dimension is {keys_shape[0]}, "
            f"expected {batch_size}"
        )
    if keys_shape[1] != transform.num_heads:
        raise ChunkTransformValidationError(
            f"correction_keys H dimension is {keys_shape[1]}, "
            f"expected num_heads={transform.num_heads}"
        )
    if keys_shape[2] != transform.chunk_size:
        raise ChunkTransformValidationError(
            f"correction_keys C dimension is {keys_shape[2]}, "
            f"expected chunk_size={transform.chunk_size}"
        )
    if keys_shape[3] != transform.key_dim:
        raise ChunkTransformValidationError(
            f"correction_keys d_k dimension is {keys_shape[3]}, "
            f"expected key_dim={transform.key_dim}"
        )

    # correction_weights: (B, H, C)
    weights_shape = transform.correction_weights.shape
    if len(weights_shape) != 3:
        raise ChunkTransformValidationError(
            f"correction_weights must be 3D (B, H, C), got shape {weights_shape}"
        )
    if weights_shape[0] != batch_size:
        raise ChunkTransformValidationError(
            f"correction_weights batch dimension is {weights_shape[0]}, "
            f"expected {batch_size}"
        )
    if weights_shape[1] != transform.num_heads:
        raise ChunkTransformValidationError(
            f"correction_weights H dimension is {weights_shape[1]}, "
            f"expected num_heads={transform.num_heads}"
        )
    if weights_shape[2] != transform.chunk_size:
        raise ChunkTransformValidationError(
            f"correction_weights C dimension is {weights_shape[2]}, "
            f"expected chunk_size={transform.chunk_size}"
        )

    # additive_term: (B, H, d_k, d_v)
    additive_shape = transform.additive_term.shape
    if len(additive_shape) != 4:
        raise ChunkTransformValidationError(
            f"additive_term must be 4D (B, H, d_k, d_v), got shape {additive_shape}"
        )
    if additive_shape[0] != batch_size:
        raise ChunkTransformValidationError(
            f"additive_term batch dimension is {additive_shape[0]}, "
            f"expected {batch_size}"
        )
    if additive_shape[1] != transform.num_heads:
        raise ChunkTransformValidationError(
            f"additive_term H dimension is {additive_shape[1]}, "
            f"expected num_heads={transform.num_heads}"
        )
    if additive_shape[2] != transform.key_dim:
        raise ChunkTransformValidationError(
            f"additive_term d_k dimension is {additive_shape[2]}, "
            f"expected key_dim={transform.key_dim}"
        )
    if additive_shape[3] != transform.value_dim:
        raise ChunkTransformValidationError(
            f"additive_term d_v dimension is {additive_shape[3]}, "
            f"expected value_dim={transform.value_dim}"
        )
