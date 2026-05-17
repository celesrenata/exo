"""Chunk-level transform representation and computation for GatedDeltaNet chunked prefill.

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


# ---------------------------------------------------------------------------
# Chunk-local computation
# ---------------------------------------------------------------------------


def compute_chunk_local(
    *,
    q_chunk: torch.Tensor,      # (B, C, H, d_k) — queries for this chunk
    k_chunk: torch.Tensor,      # (B, C, H, d_k) — keys for this chunk
    v_chunk: torch.Tensor,      # (B, C, H, d_v) — values for this chunk
    gate_chunk: torch.Tensor,   # (B, C, H) — log-space decay gates
    beta_chunk: torch.Tensor,   # (B, C, H) — sigmoid update rates
    chunk_index: int,
) -> tuple[ChunkOutput, ChunkTransform]:
    """Compute intra-chunk outputs and the chunk's composed transform.

    This function processes all tokens within a single chunk:
    1. Computes the cumulative log-decay within the chunk
    2. Builds the decay-weighted causal attention matrix L[i,j] = exp(G[i] - G[j])
    3. Computes intra-chunk outputs using the causal attention
    4. Builds the ChunkTransform representing the composed affine transform

    The intra-chunk outputs do NOT include the contribution from the incoming
    state (which is unknown during parallel chunk processing). That contribution
    is added during the output materialization phase.

    All computation is done in fp32 for numerical stability.

    Args:
        q_chunk: L2-normalized queries, shape (B, C, H, d_k)
        k_chunk: L2-normalized keys, shape (B, C, H, d_k)
        v_chunk: values, shape (B, C, H, d_v)
        gate_chunk: log-space decay gates, shape (B, C, H)
        beta_chunk: sigmoid update rates, shape (B, C, H)
        chunk_index: zero-based index of this chunk within the sequence

    Returns:
        Tuple of (ChunkOutput, ChunkTransform):
        - ChunkOutput contains intra-chunk activations of shape (B, C, H, d_v)
        - ChunkTransform contains the composed affine transform for the chunk

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    batch_size, chunk_size, num_heads, key_dim = q_chunk.shape
    value_dim = v_chunk.shape[-1]

    # Cast all inputs to fp32 for numerical stability
    q = q_chunk.float()
    k = k_chunk.float()
    v = v_chunk.float()
    gate = gate_chunk.float()
    beta = beta_chunk.float()

    # Scale queries: scale = 1/sqrt(d_k)
    scale = key_dim ** -0.5
    q = q * scale

    # -----------------------------------------------------------------------
    # Compute cumulative log-decay within the chunk
    # G[i] = sum(gate[0:i+1]) — cumulative sum of log-decays
    # gate is in log-space (negative values), so exp(G[i]) is the total decay
    # from the start of the chunk to position i.
    # -----------------------------------------------------------------------
    # gate shape: (B, C, H)
    cumulative_log_decay = gate.cumsum(dim=1)  # (B, C, H)

    # -----------------------------------------------------------------------
    # Sequential intra-chunk computation for correctness
    #
    # We iterate over tokens in the chunk, maintaining a local state that
    # starts at zero (the incoming state contribution is handled separately
    # during output materialization).
    #
    # For each token t in [0, C-1]:
    #   local_state = exp(g_t) * local_state + beta_t * k_t ⊗ (v_t - exp(g_t) * local_state^T k_t)
    #   o_intra[t] = q_t^T @ local_state
    #
    # This matches the recurrence in _gated_deltanet_recurrent_step_impl but
    # with initial state = 0 (the inter-chunk state contribution is added later).
    # -----------------------------------------------------------------------

    # Initialize local state to zero — shape (B, H, d_k, d_v)
    local_state = torch.zeros(
        batch_size, num_heads, key_dim, value_dim,
        dtype=torch.float32, device=q.device,
    )

    # Output activations — shape (B, C, H, d_v)
    outputs = torch.zeros(
        batch_size, chunk_size, num_heads, value_dim,
        dtype=torch.float32, device=q.device,
    )

    # Correction keys for the WY representation
    # correction_keys: (B, H, C, d_k) — key vectors at each position
    correction_keys = torch.zeros(
        batch_size, num_heads, chunk_size, key_dim,
        dtype=torch.float32, device=q.device,
    )

    for t in range(chunk_size):
        # Extract token-level tensors
        q_t = q[:, t, :, :]       # (B, H, d_k)
        k_t = k[:, t, :, :]       # (B, H, d_k)
        v_t = v[:, t, :, :]       # (B, H, d_v)
        g_t = gate[:, t, :]       # (B, H)
        b_t = beta[:, t, :]       # (B, H)

        # Step 1: Decay local state
        decay = g_t.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        local_state = local_state * decay  # (B, H, d_k, d_v)

        # Step 2: Retrieve what state predicts for this key
        k_expanded = k_t.unsqueeze(-1)  # (B, H, d_k, 1)
        retrieved = (local_state * k_expanded).sum(dim=-2)  # (B, H, d_v)

        # Step 3: Compute delta (error correction)
        beta_expanded = b_t.unsqueeze(-1)  # (B, H, 1)
        delta = beta_expanded * (v_t - retrieved)  # (B, H, d_v)

        # Step 4: Write correction into state (outer product update)
        delta_expanded = delta.unsqueeze(-2)  # (B, H, 1, d_v)
        local_state = local_state + k_expanded * delta_expanded  # (B, H, d_k, d_v)

        # Step 5: Read output (scale already applied to q)
        q_expanded = q_t.unsqueeze(-1)  # (B, H, d_k, 1)
        output_t = (local_state * q_expanded).sum(dim=-2)  # (B, H, d_v)

        # Store output: outputs is (B, C, H, d_v)
        outputs[:, t, :, :] = output_t

        # Store correction keys and weights for WY representation
        # correction_keys[b, h, t, :] = k_t[b, h, :]
        correction_keys[:, :, t, :] = k_t

        # correction_weights[b, h, t] = beta_t * exp(G_total - G_t)
        # This is the relative decay from position t to the end of the chunk.
        # We compute this after the loop using cumulative_log_decay.

    # -----------------------------------------------------------------------
    # Build ChunkTransform
    # -----------------------------------------------------------------------

    # Total cumulative log-decay across the chunk: sum of all gates
    # cumulative_log_decay[:, -1, :] is G[C-1] = sum(gate[0:C])
    total_log_decay = cumulative_log_decay[:, -1, :]  # (B, H)

    # Correction weights: beta_t * exp(G_total - G_t)
    # This represents how much each token's correction contributes to the
    # final composed transform, accounting for decay from that position to
    # the end of the chunk.
    # G_total shape: (B, 1, H), cumulative_log_decay shape: (B, C, H)
    relative_decay_to_end = (
        total_log_decay.unsqueeze(1) - cumulative_log_decay
    ).exp()  # (B, C, H)

    # correction_weights = beta * relative_decay_to_end
    # beta shape: (B, C, H), relative_decay_to_end shape: (B, C, H)
    correction_weights_final = beta * relative_decay_to_end  # (B, C, H)
    # Permute to (B, H, C) for the ChunkTransform
    correction_weights_final = correction_weights_final.permute(0, 2, 1)  # (B, H, C)

    # Additive term: the accumulated B terms with decay
    # B_composed = sum_t(exp(G_total - G_t) * beta_t * k_t ⊗ v_t)
    # This is the local_state at the end of the chunk (which started from zero).
    additive_term = local_state  # (B, H, d_k, d_v)

    chunk_transform = ChunkTransform(
        cumulative_log_decay=total_log_decay,
        correction_keys=correction_keys,
        correction_weights=correction_weights_final,
        additive_term=additive_term,
        chunk_size=chunk_size,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )

    chunk_output = ChunkOutput(
        activations=outputs,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
    )

    return chunk_output, chunk_transform
