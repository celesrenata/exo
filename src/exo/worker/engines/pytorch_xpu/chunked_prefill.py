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

import logging
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch

if TYPE_CHECKING:
    from exo.worker.engines.pytorch_xpu.instrumentation import PerformanceRecorder

logger = logging.getLogger(__name__)

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
        - correction_core: dense WY core matrix (C × C), block-lower-triangular after composition
        - additive_term: B_composed (d_k × d_v) — the accumulated additive state

    All tensors are fp32 for numerical stability.

    The WY representation encodes A_composed implicitly as:
        A_composed(S) = exp(cumulative_log_decay) * (S - K^T @ T @ (K @ S))

    where K = correction_keys and T = correction_core.

    For a single chunk, T = diag(correction_weights). After composing
    multiple chunks, T becomes block-lower-triangular. This representation
    avoids materializing the full (d_k × d_k) matrix while supporting
    efficient application and associative composition.
    """

    cumulative_log_decay: torch.Tensor
    """Total log-decay across the chunk, shape (B, H). Sum of per-token log-gates."""

    correction_keys: torch.Tensor
    """Accumulated key vectors for WY rank-1 corrections, shape (B, H, C, d_k).

    Each row is a key vector from a token in the chunk, used to represent
    the rank-1 corrections to the identity in the WY decomposition.
    """

    correction_core: torch.Tensor
    """Dense WY core matrix, shape (B, H, C, C).

    For a single chunk, this is diag(correction_weights) — a diagonal matrix
    where each diagonal entry combines the token's β value and relative decay
    factor within the chunk.

    After composition of multiple chunks, this becomes block-lower-triangular:
        T_composed = [[T1,    0   ],
                      [cross, T2  ]]
    where cross = -T2 @ (K2 @ K1^T) @ T1.

    The WY representation encodes A_composed implicitly as:
        A_composed(S) = exp(cumulative_log_decay) * (S - K^T @ T @ (K @ S))
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
        4. correction_core has shape (B, H, C, C) matching chunk_size
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
    if transform.correction_core.dtype != torch.float32:
        raise ChunkTransformValidationError(
            f"correction_core must be fp32, got {transform.correction_core.dtype}"
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

    # correction_core: (B, H, C, C)
    core_shape = transform.correction_core.shape
    if len(core_shape) != 4:
        raise ChunkTransformValidationError(
            f"correction_core must be 4D (B, H, C, C), got shape {core_shape}"
        )
    if core_shape[0] != batch_size:
        raise ChunkTransformValidationError(
            f"correction_core batch dimension is {core_shape[0]}, "
            f"expected {batch_size}"
        )
    if core_shape[1] != transform.num_heads:
        raise ChunkTransformValidationError(
            f"correction_core H dimension is {core_shape[1]}, "
            f"expected num_heads={transform.num_heads}"
        )
    if core_shape[2] != transform.chunk_size:
        raise ChunkTransformValidationError(
            f"correction_core row dimension is {core_shape[2]}, "
            f"expected chunk_size={transform.chunk_size}"
        )
    if core_shape[3] != transform.chunk_size:
        raise ChunkTransformValidationError(
            f"correction_core column dimension is {core_shape[3]}, "
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

    # Build the dense correction core matrix: diag(correction_weights)
    # For a single chunk, the core is diagonal — shape (B, H, C, C)
    correction_core = torch.diag_embed(correction_weights_final)  # (B, H, C, C)

    # Additive term: the accumulated B terms with decay
    # B_composed = sum_t(exp(G_total - G_t) * beta_t * k_t ⊗ v_t)
    # This is the local_state at the end of the chunk (which started from zero).
    additive_term = local_state  # (B, H, d_k, d_v)

    chunk_transform = ChunkTransform(
        cumulative_log_decay=total_log_decay,
        correction_keys=correction_keys,
        correction_core=correction_core,
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


# ---------------------------------------------------------------------------
# Chunk transform application and composition
# ---------------------------------------------------------------------------


def apply_chunk_transform(
    incoming_state: torch.Tensor,
    transform: ChunkTransform,
) -> torch.Tensor:
    """Apply a chunk's composed transform to an incoming state.

    Computes: S_out = A_composed * S_in + B_composed

    Using the WY representation with dense core matrix T:
        S_out = exp(decay) * (S_in - K^T @ T @ (K @ S_in)) + additive_term

    The correction term accounts for the rank-1 perturbations accumulated
    across all tokens in the chunk. The dense core matrix T encodes the
    coupling between corrections (diagonal for single chunks, block-lower-
    triangular after composition).

    All computation is done in fp32 for numerical stability.

    Args:
        incoming_state: State from previous chunk, shape (B, H, d_k, d_v), fp32.
        transform: The ChunkTransform representing the chunk's composed affine transform.

    Returns:
        Output state after applying the transform, shape (B, H, d_k, d_v), fp32.

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    # Ensure fp32 computation
    state = incoming_state.float()

    # Extract transform components
    decay = transform.cumulative_log_decay  # (B, H)
    keys = transform.correction_keys       # (B, H, C, d_k)
    core = transform.correction_core       # (B, H, C, C)
    additive = transform.additive_term     # (B, H, d_k, d_v)

    # Compute corrections using the dense core matrix T:
    # Step 1: K @ S_in — project state onto key directions
    # keys: (B, H, C, d_k), state: (B, H, d_k, d_v) -> (B, H, C, d_v)
    projected = torch.einsum("bhck,bhkv->bhcv", keys, state)

    # Step 2: T @ (K @ S_in) — apply core matrix in correction space
    # core: (B, H, C, C), projected: (B, H, C, d_v) -> (B, H, C, d_v)
    weighted = torch.einsum("bhij,bhjv->bhiv", core, projected)

    # Step 3: K^T @ T @ (K @ S_in) — project back to state space
    # keys: (B, H, C, d_k), weighted: (B, H, C, d_v) -> (B, H, d_k, d_v)
    correction = torch.einsum("bhck,bhcv->bhkv", keys, weighted)

    # Apply decay and subtract corrections, then add the additive term
    # S_out = exp(decay) * (S_in - correction) + additive_term
    decay_factor = decay.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    output_state = decay_factor * (state - correction) + additive

    return output_state


def apply_transform_linear(
    transform: ChunkTransform,
    state: torch.Tensor,
) -> torch.Tensor:
    """Apply only the linear part A of a chunk transform to a state matrix.

    Computes: A(S) = exp(decay) * (S - K^T @ T @ (K @ S))

    This is the same as apply_chunk_transform but WITHOUT the additive term.
    Used during transform composition to compute A_2(B_1).

    All computation is done in fp32 for numerical stability.

    Args:
        transform: The ChunkTransform whose linear part to apply.
        state: State matrix to transform, shape (B, H, d_k, d_v), fp32.

    Returns:
        Transformed state, shape (B, H, d_k, d_v), fp32.

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    state = state.float()

    decay = transform.cumulative_log_decay  # (B, H)
    keys = transform.correction_keys       # (B, H, C, d_k)
    core = transform.correction_core       # (B, H, C, C)

    # K @ S: project state onto key directions
    projected = torch.einsum("bhck,bhkv->bhcv", keys, state)  # (B, H, C, d_v)

    # T @ (K @ S): apply core matrix
    weighted = torch.einsum("bhij,bhjv->bhiv", core, projected)  # (B, H, C, d_v)

    # K^T @ T @ (K @ S): project back to state space
    correction = torch.einsum("bhck,bhcv->bhkv", keys, weighted)  # (B, H, d_k, d_v)

    # A(S) = exp(decay) * (S - correction)
    decay_factor = decay.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    return decay_factor * (state - correction)


def compose_two_transforms(
    t1: ChunkTransform,
    t2: ChunkTransform,
) -> ChunkTransform:
    """Compose two chunk transforms: apply t1 first, then t2.

    The composition of affine transforms (A_1, B_1) then (A_2, B_2) is:
        A_composed = A_2 ∘ A_1
        B_composed = A_2(B_1) + B_2

    The composed WY representation uses block-structured core:
        K_composed = [K1; K2]  (concatenated along C dimension)
        T_composed = [[T1,    0   ],
                      [cross, T2  ]]
        where cross = -T2 @ (K2 @ K1^T) @ T1

    This preserves the WY form and enables further composition.

    All computation is done in fp32 for numerical stability.

    Args:
        t1: First transform to apply (earlier in sequence).
        t2: Second transform to apply (later in sequence).

    Returns:
        Composed ChunkTransform representing t2 ∘ t1.

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    # 1. Composed log-decay: sum of individual decays
    decay_composed = t1.cumulative_log_decay + t2.cumulative_log_decay  # (B, H)

    # 2. Composed keys: concatenate along C dimension
    keys_composed = torch.cat([t1.correction_keys, t2.correction_keys], dim=-2)  # (B, H, C1+C2, d_k)

    # 3. Composed core: block-lower-triangular matrix
    # cross = -T2 @ (K2 @ K1^T) @ T1
    # K2 @ K1^T: (B, H, C2, d_k) @ (B, H, d_k, C1) -> (B, H, C2, C1)
    k2_k1t = torch.einsum("bhck,bhdk->bhcd", t2.correction_keys, t1.correction_keys)  # (B, H, C2, C1)

    # (K2 @ K1^T) @ T1: (B, H, C2, C1) @ (B, H, C1, C1) -> (B, H, C2, C1)
    k2k1t_t1 = torch.einsum("bhij,bhjk->bhik", k2_k1t, t1.correction_core)  # (B, H, C2, C1)

    # T2 @ (K2 @ K1^T) @ T1: (B, H, C2, C2) @ (B, H, C2, C1) -> (B, H, C2, C1)
    cross = -torch.einsum("bhij,bhjk->bhik", t2.correction_core, k2k1t_t1)  # (B, H, C2, C1)

    # Build block matrix: [[T1, 0], [cross, T2]]
    batch_size = t1.cumulative_log_decay.shape[0]
    num_heads = t1.num_heads
    c1 = t1.chunk_size
    c2 = t2.chunk_size
    c_total = c1 + c2
    device = t1.correction_core.device

    core_composed = torch.zeros(
        batch_size, num_heads, c_total, c_total,
        dtype=torch.float32, device=device,
    )
    # Top-left: T1
    core_composed[:, :, :c1, :c1] = t1.correction_core
    # Bottom-left: cross
    core_composed[:, :, c1:, :c1] = cross
    # Bottom-right: T2
    core_composed[:, :, c1:, c1:] = t2.correction_core

    # 4. Composed additive: A_2(B_1) + B_2
    additive_composed = apply_transform_linear(t2, t1.additive_term) + t2.additive_term

    return ChunkTransform(
        cumulative_log_decay=decay_composed,
        correction_keys=keys_composed,
        correction_core=core_composed,
        additive_term=additive_composed,
        chunk_size=c_total,
        num_heads=num_heads,
        key_dim=t1.key_dim,
        value_dim=t1.value_dim,
    )


def compose_chunk_transforms(
    transforms: list[ChunkTransform],
) -> list[torch.Tensor]:
    """Compose chunk transforms sequentially to compute prefix states.

    Given N chunk transforms, computes the incoming state for each chunk
    by sequentially applying transforms starting from zero state.

    The composition is associative: applying transform_i to state_i produces
    state_{i+1}. This function computes the full prefix scan:
        state_0 = 0
        state_1 = apply_chunk_transform(state_0, transform_0)
        state_2 = apply_chunk_transform(state_1, transform_1)
        ...
        state_N = apply_chunk_transform(state_{N-1}, transform_{N-1})

    Returns N+1 states: the initial zero state plus the output state after
    each chunk transform. state_i is the incoming state for chunk i (for
    i < N), and state_N is the final recurrent state after all chunks.

    All computation is done in fp32 for numerical stability.

    Args:
        transforms: List of N ChunkTransforms, one per chunk in sequence order.

    Returns:
        List of N+1 state tensors, each shape (B, H, d_k, d_v) in fp32.
        states[0] is the zero initial state, states[i] for i >= 1 is the
        state after applying transforms[0] through transforms[i-1].

    Raises:
        ValueError: If transforms list is empty.

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    if not transforms:
        raise ValueError("transforms list must not be empty")

    # Infer dimensions from the first transform
    first = transforms[0]
    batch_size = first.cumulative_log_decay.shape[0]
    num_heads = first.num_heads
    key_dim = first.key_dim
    value_dim = first.value_dim
    device = first.additive_term.device

    # Initial state is zero
    zero_state = torch.zeros(
        batch_size, num_heads, key_dim, value_dim,
        dtype=torch.float32, device=device,
    )

    # Compute prefix states sequentially
    states: list[torch.Tensor] = [zero_state]
    current_state = zero_state

    for transform in transforms:
        current_state = apply_chunk_transform(current_state, transform)
        states.append(current_state)

    return states


# ---------------------------------------------------------------------------
# Output materialization
# ---------------------------------------------------------------------------


def materialize_chunk_outputs(
    *,
    chunk_outputs: list[ChunkOutput],
    prefix_states: list[torch.Tensor],
    q_chunks: list[torch.Tensor],
    k_chunks: list[torch.Tensor],
    gate_chunks: list[torch.Tensor],
    beta_chunks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize final per-token outputs and the final recurrent state.

    For each chunk c with incoming state S_c (from prefix_states[c]):
    - The inter-chunk contribution to token t in chunk c accounts for how
      the incoming state S_c propagates through the recurrence within the chunk.
    - The final output is: o_final[t] = o_intra[t] + o_inter[t]

    The inter-chunk contribution is computed by running the linear part of the
    recurrence on the incoming state. For token t at position p within chunk c:
        The state contribution from S_c at position p is obtained by applying
        the partial linear transform (decay + corrections from tokens 0..p) to S_c.
        Then o_inter[t] = q_t^T @ A_partial_t(S_c)

    For exact correctness, we re-run the recurrence sequentially within each chunk
    using the actual incoming state and subtract the zero-state contribution
    (which is already captured in chunk_outputs).

    Args:
        chunk_outputs: List of N ChunkOutput objects with intra-chunk activations.
        prefix_states: List of N+1 state tensors from compose_chunk_transforms.
            prefix_states[c] is the incoming state for chunk c.
            prefix_states[N] is the final state after all chunks.
        q_chunks: List of N query tensors, each (B, C, H, d_k).
        k_chunks: List of N key tensors, each (B, C, H, d_k).
        gate_chunks: List of N gate tensors, each (B, C, H).
        beta_chunks: List of N beta tensors, each (B, C, H).

    Returns:
        (outputs, final_state) where:
        - outputs: (B, T, H, d_v) — final per-token outputs in sequence order
        - final_state: (B, H, d_k, d_v) — final recurrent state for decode continuation

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    num_chunks = len(chunk_outputs)
    assert num_chunks > 0, "Must have at least one chunk"
    assert len(prefix_states) == num_chunks + 1, (
        f"Expected {num_chunks + 1} prefix states, got {len(prefix_states)}"
    )

    # Infer dimensions from first chunk
    first_output = chunk_outputs[0]
    batch_size = first_output.activations.shape[0]
    num_heads = first_output.activations.shape[2]
    value_dim = first_output.activations.shape[3]
    key_dim = q_chunks[0].shape[-1]
    device = first_output.activations.device

    # Compute total sequence length
    total_seq_len = sum(co.chunk_size for co in chunk_outputs)

    # Allocate output tensor
    outputs = torch.zeros(
        batch_size, total_seq_len, num_heads, value_dim,
        dtype=torch.float32, device=device,
    )

    # For each chunk, compute the inter-chunk contribution and add to intra-chunk output
    offset = 0
    final_state = prefix_states[-1]  # Default: use the last prefix state

    for c in range(num_chunks):
        chunk_size = chunk_outputs[c].chunk_size
        incoming_state = prefix_states[c]  # (B, H, d_k, d_v)
        intra_output = chunk_outputs[c].activations  # (B, C, H, d_v)

        q_chunk = q_chunks[c].float()  # (B, C, H, d_k)
        k_chunk = k_chunks[c].float()  # (B, C, H, d_k)
        gate_chunk = gate_chunks[c].float()  # (B, C, H)
        beta_chunk = beta_chunks[c].float()  # (B, C, H)

        # Scale queries
        scale = key_dim ** -0.5
        q_scaled = q_chunk * scale

        # Compute inter-chunk contribution by propagating incoming_state
        # through the recurrence within this chunk.
        # We track how S_c evolves through the chunk's tokens (linear part only).
        state_contribution = incoming_state.float()  # (B, H, d_k, d_v)

        for t in range(chunk_size):
            q_t = q_scaled[:, t, :, :]   # (B, H, d_k)
            k_t = k_chunk[:, t, :, :]    # (B, H, d_k)
            g_t = gate_chunk[:, t, :]    # (B, H)
            b_t = beta_chunk[:, t, :]    # (B, H)

            # Decay the state contribution
            decay = g_t.exp().unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            state_contribution = state_contribution * decay

            # Apply the delta rule correction to the state contribution
            # The correction erases the component along k_t proportional to beta_t
            k_expanded = k_t.unsqueeze(-1)  # (B, H, d_k, 1)
            retrieved = (state_contribution * k_expanded).sum(dim=-2)  # (B, H, d_v)
            beta_expanded = b_t.unsqueeze(-1)  # (B, H, 1)
            correction = beta_expanded * retrieved  # (B, H, d_v)
            correction_expanded = correction.unsqueeze(-2)  # (B, H, 1, d_v)
            state_contribution = state_contribution - k_expanded * correction_expanded

            # Inter-chunk output contribution: q_t^T @ state_contribution
            q_expanded = q_t.unsqueeze(-1)  # (B, H, d_k, 1)
            o_inter_t = (state_contribution * q_expanded).sum(dim=-2)  # (B, H, d_v)

            # Final output = intra + inter
            outputs[:, offset + t, :, :] = intra_output[:, t, :, :] + o_inter_t

        offset += chunk_size

    # The final recurrent state is the last prefix state (state after all chunks)
    # This already accounts for the full recurrence including the incoming state
    # propagation through all chunks.
    return outputs, final_state


# ---------------------------------------------------------------------------
# Top-level chunked prefill orchestration
# ---------------------------------------------------------------------------


def chunked_gated_deltanet_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full chunked GatedDeltaNet prefill.

    Splits the sequence into chunks, computes intra-chunk outputs and transforms,
    propagates prefix states, and materializes final outputs.

    Returns (outputs, final_state) matching the sequential prefill interface.

    Args:
        q: L2-normalized queries, shape (B, T, H, d_k).
        k: L2-normalized keys, shape (B, T, H, d_k).
        v: Values, shape (B, T, H, d_v).
        gate: Log-space decay gates, shape (B, T, H).
        beta: Sigmoid update rates, shape (B, T, H).
        chunk_size: Number of tokens per chunk (default 64).
        initial_state: Optional initial recurrent state, shape (B, H, d_k, d_v).
            If None, starts from zero state.

    Returns:
        (outputs, final_state) where:
        - outputs: (B, T, H, d_v) — per-token outputs in sequence order
        - final_state: (B, H, d_k, d_v) — final recurrent state for decode

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    batch_size, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    device = q.device

    # Phase 1: Split sequence into chunks and compute chunk-local outputs/transforms
    chunk_outputs: list[ChunkOutput] = []
    transforms: list[ChunkTransform] = []
    q_chunks: list[torch.Tensor] = []
    k_chunks: list[torch.Tensor] = []
    gate_chunks: list[torch.Tensor] = []
    beta_chunks: list[torch.Tensor] = []

    offset = 0
    chunk_index = 0
    while offset < seq_len:
        end = min(offset + chunk_size, seq_len)
        c_size = end - offset

        q_c = q[:, offset:end]
        k_c = k[:, offset:end]
        v_c = v[:, offset:end]
        gate_c = gate[:, offset:end]
        beta_c = beta[:, offset:end]

        chunk_out, chunk_transform = compute_chunk_local(
            q_chunk=q_c,
            k_chunk=k_c,
            v_chunk=v_c,
            gate_chunk=gate_c,
            beta_chunk=beta_c,
            chunk_index=chunk_index,
        )

        chunk_outputs.append(chunk_out)
        transforms.append(chunk_transform)
        q_chunks.append(q_c)
        k_chunks.append(k_c)
        gate_chunks.append(gate_c)
        beta_chunks.append(beta_c)

        offset = end
        chunk_index += 1

    # Phase 2: Compose transforms to get prefix states
    prefix_states = compose_chunk_transforms(transforms)

    # If initial_state is provided, shift all prefix states by applying
    # the initial state through the transform chain
    if initial_state is not None:
        initial = initial_state.float()
        # Recompute prefix states starting from initial_state instead of zero
        # prefix_states[0] should be initial_state
        # prefix_states[i] = apply_chunk_transform(prefix_states[i-1], transforms[i-1])
        shifted_states: list[torch.Tensor] = [initial]
        current = initial
        for transform in transforms:
            current = apply_chunk_transform(current, transform)
            shifted_states.append(current)
        prefix_states = shifted_states

    # Phase 3: Materialize final outputs using prefix states
    outputs, final_state = materialize_chunk_outputs(
        chunk_outputs=chunk_outputs,
        prefix_states=prefix_states,
        q_chunks=q_chunks,
        k_chunks=k_chunks,
        gate_chunks=gate_chunks,
        beta_chunks=beta_chunks,
    )

    return outputs, final_state

# ---------------------------------------------------------------------------
# Fallback logic (Subtask 7)
# ---------------------------------------------------------------------------

# Supported dtypes for chunked prefill computation
_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset({
    torch.float32,
    torch.float16,
    torch.bfloat16,
})


class ChunkedPrefillUnsupportedError(ValueError):
    """Raised when chunked prefill encounters an unsupported configuration.

    This structured error is raised when ``fallback_on_unsupported_shape=False``
    and the input dtype, shape, or parameter combination is not supported by
    the chunked prefill implementation.

    Attributes:
        reason: Human-readable description of why the configuration is unsupported.
        dtype: The unsupported dtype (if dtype was the issue).
        shape: The unsupported shape (if shape was the issue).
    """

    def __init__(
        self,
        reason: str,
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.reason = reason
        self.dtype = dtype
        self.shape = shape
        super().__init__(reason)


class ChunkedPrefillFallbackWarning(UserWarning):
    """Warning emitted when chunked prefill falls back to sequential.

    This structured warning is emitted when ``fallback_on_unsupported_shape=True``
    and the input configuration is not supported, causing a fallback to
    sequential prefill.
    """


def _validate_chunked_prefill_inputs(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> str | None:
    """Validate inputs for chunked prefill support.

    Returns None if inputs are supported, or a reason string if unsupported.
    """
    # Check sequence length
    if q.shape[1] < 1:
        return f"Sequence length must be >= 1, got {q.shape[1]}"

    # Check dtypes
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.dtype not in _SUPPORTED_DTYPES:
            return (
                f"Unsupported dtype for {name}: {tensor.dtype}. "
                f"Supported: {sorted(str(d) for d in _SUPPORTED_DTYPES)}"
            )

    # Check gate and beta dtypes
    if gate.dtype not in _SUPPORTED_DTYPES:
        return f"Unsupported dtype for gate: {gate.dtype}"
    if beta.dtype not in _SUPPORTED_DTYPES:
        return f"Unsupported dtype for beta: {beta.dtype}"

    # Check shape consistency
    if q.ndim != 4:
        return f"q must be 4D (B, T, H, d_k), got {q.ndim}D"
    if k.ndim != 4:
        return f"k must be 4D (B, T, H, d_k), got {k.ndim}D"
    if v.ndim != 4:
        return f"v must be 4D (B, T, H, d_v), got {v.ndim}D"
    if gate.ndim != 3:
        return f"gate must be 3D (B, T, H), got {gate.ndim}D"
    if beta.ndim != 3:
        return f"beta must be 3D (B, T, H), got {beta.ndim}D"

    # Check batch/seq/head consistency
    batch_size, seq_len, num_heads, key_dim = q.shape
    if k.shape[:3] != (batch_size, seq_len, num_heads):
        return f"k shape {k.shape} inconsistent with q shape {q.shape}"
    if v.shape[:3] != (batch_size, seq_len, num_heads):
        return f"v shape {v.shape} inconsistent with q shape {q.shape}"
    if gate.shape != (batch_size, seq_len, num_heads):
        return f"gate shape {gate.shape} inconsistent with q shape {q.shape}"
    if beta.shape != (batch_size, seq_len, num_heads):
        return f"beta shape {beta.shape} inconsistent with q shape {q.shape}"

    # Check chunk_size validity
    if chunk_size < 1:
        return f"chunk_size must be >= 1, got {chunk_size}"

    return None


def _sequential_prefill_fallback(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential prefill fallback — processes tokens one at a time.

    This is the safe fallback path used when chunked prefill cannot handle
    the input configuration. It runs the full sequential recurrence.

    Returns (outputs, final_state) matching the chunked prefill interface.
    """
    batch_size, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]

    q_fp32 = q.float()
    k_fp32 = k.float()
    v_fp32 = v.float()
    gate_fp32 = gate.float()
    beta_fp32 = beta.float()

    scale = key_dim ** -0.5
    q_fp32 = q_fp32 * scale

    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = torch.zeros(
            batch_size, num_heads, key_dim, value_dim,
            dtype=torch.float32, device=q.device,
        )

    outputs = torch.zeros(
        batch_size, seq_len, num_heads, value_dim,
        dtype=torch.float32, device=q.device,
    )

    for t in range(seq_len):
        q_t = q_fp32[:, t, :, :]
        k_t = k_fp32[:, t, :, :]
        v_t = v_fp32[:, t, :, :]
        g_t = gate_fp32[:, t, :]
        b_t = beta_fp32[:, t, :]

        decay = g_t.exp().unsqueeze(-1).unsqueeze(-1)
        state = state * decay

        k_expanded = k_t.unsqueeze(-1)
        retrieved = (state * k_expanded).sum(dim=-2)

        beta_expanded = b_t.unsqueeze(-1)
        delta = beta_expanded * (v_t - retrieved)

        delta_expanded = delta.unsqueeze(-2)
        state = state + k_expanded * delta_expanded

        q_expanded = q_t.unsqueeze(-1)
        output_t = (state * q_expanded).sum(dim=-2)
        outputs[:, t, :, :] = output_t

    return outputs, state


def chunked_prefill_with_fallback(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    fallback_on_unsupported_shape: bool = True,
    performance_recorder: PerformanceRecorder | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked GatedDeltaNet prefill with fallback logic.

    Validates inputs and either runs chunked prefill or falls back to
    sequential prefill depending on the ``fallback_on_unsupported_shape`` flag.

    When inputs are unsupported:
    - If ``fallback_on_unsupported_shape=True``: emits a structured warning
      and uses sequential prefill.
    - If ``fallback_on_unsupported_shape=False``: raises
      ``ChunkedPrefillUnsupportedError``.

    Args:
        q: L2-normalized queries, shape (B, T, H, d_k).
        k: L2-normalized keys, shape (B, T, H, d_k).
        v: Values, shape (B, T, H, d_v).
        gate: Log-space decay gates, shape (B, T, H).
        beta: Sigmoid update rates, shape (B, T, H).
        chunk_size: Number of tokens per chunk (default 64).
        initial_state: Optional initial recurrent state, shape (B, H, d_k, d_v).
        fallback_on_unsupported_shape: Whether to fall back to sequential
            prefill on unsupported inputs (True) or raise an error (False).
        performance_recorder: Optional recorder for instrumentation spans.

    Returns:
        (outputs, final_state) where:
        - outputs: (B, T, H, d_v) — per-token outputs in sequence order
        - final_state: (B, H, d_k, d_v) — final recurrent state for decode

    Raises:
        ChunkedPrefillUnsupportedError: When inputs are unsupported and
            ``fallback_on_unsupported_shape=False``.

    **Validates: Requirements 7.1, 7.5, 7.6, 7.7**
    """
    # Validate inputs
    unsupported_reason = _validate_chunked_prefill_inputs(
        q=q, k=k, v=v, gate=gate, beta=beta, chunk_size=chunk_size,
    )

    if unsupported_reason is not None:
        if fallback_on_unsupported_shape:
            warnings.warn(
                f"Chunked prefill unsupported: {unsupported_reason}. "
                f"Falling back to sequential prefill.",
                ChunkedPrefillFallbackWarning,
                stacklevel=2,
            )
            logger.warning(
                "Chunked prefill fallback to sequential: %s",
                unsupported_reason,
            )
            if performance_recorder is not None:
                with performance_recorder.span(
                    "chunked_prefill_sequential_fallback",
                    mode="prefill",
                    metadata={"reason": unsupported_reason},
                ):
                    return _sequential_prefill_fallback(
                        q=q, k=k, v=v, gate=gate, beta=beta,
                        initial_state=initial_state,
                    )
            return _sequential_prefill_fallback(
                q=q, k=k, v=v, gate=gate, beta=beta,
                initial_state=initial_state,
            )
        else:
            raise ChunkedPrefillUnsupportedError(
                unsupported_reason,
                dtype=q.dtype if q.dtype not in _SUPPORTED_DTYPES else None,
                shape=tuple(q.shape) if q.ndim != 4 else None,
            )

    # Inputs are valid — run instrumented chunked prefill
    if performance_recorder is not None:
        return _chunked_prefill_instrumented(
            q=q, k=k, v=v, gate=gate, beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            recorder=performance_recorder,
        )

    return chunked_gated_deltanet_prefill(
        q=q, k=k, v=v, gate=gate, beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Instrumented chunked prefill (Subtask 9)
# ---------------------------------------------------------------------------


def _chunked_prefill_instrumented(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None,
    recorder: PerformanceRecorder,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked prefill with performance instrumentation.

    Records timing spans for:
    - chunk_local_computation: Time spent computing intra-chunk outputs/transforms
    - transform_composition: Time spent composing transforms into prefix states
    - output_materialization: Time spent materializing final outputs
    - chunked_prefill_total: Total end-to-end time

    Also records a sequential_prefill_comparison span for comparison timing.

    Args:
        q, k, v, gate, beta: Input tensors (same as chunked_gated_deltanet_prefill).
        chunk_size: Tokens per chunk.
        initial_state: Optional initial state.
        recorder: PerformanceRecorder to record spans into.

    Returns:
        (outputs, final_state) matching the chunked prefill interface.
    """
    batch_size, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]

    with recorder.span("chunked_prefill_total", mode="prefill", metadata={
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "num_chunks": (seq_len + chunk_size - 1) // chunk_size,
    }):
        # Phase 1: Chunk-local computation
        chunk_outputs: list[ChunkOutput] = []
        transforms: list[ChunkTransform] = []
        q_chunks: list[torch.Tensor] = []
        k_chunks: list[torch.Tensor] = []
        gate_chunks: list[torch.Tensor] = []
        beta_chunks: list[torch.Tensor] = []

        with recorder.span("chunk_local_computation", mode="prefill"):
            offset = 0
            chunk_index = 0
            while offset < seq_len:
                end = min(offset + chunk_size, seq_len)

                q_c = q[:, offset:end]
                k_c = k[:, offset:end]
                v_c = v[:, offset:end]
                gate_c = gate[:, offset:end]
                beta_c = beta[:, offset:end]

                chunk_out, chunk_transform = compute_chunk_local(
                    q_chunk=q_c,
                    k_chunk=k_c,
                    v_chunk=v_c,
                    gate_chunk=gate_c,
                    beta_chunk=beta_c,
                    chunk_index=chunk_index,
                )

                chunk_outputs.append(chunk_out)
                transforms.append(chunk_transform)
                q_chunks.append(q_c)
                k_chunks.append(k_c)
                gate_chunks.append(gate_c)
                beta_chunks.append(beta_c)

                offset = end
                chunk_index += 1

        # Phase 2: Transform composition
        with recorder.span("transform_composition", mode="prefill"):
            prefix_states = compose_chunk_transforms(transforms)

            if initial_state is not None:
                initial = initial_state.float()
                shifted_states: list[torch.Tensor] = [initial]
                current = initial
                for transform in transforms:
                    current = apply_chunk_transform(current, transform)
                    shifted_states.append(current)
                prefix_states = shifted_states

        # Phase 3: Output materialization
        with recorder.span("output_materialization", mode="prefill"):
            outputs, final_state = materialize_chunk_outputs(
                chunk_outputs=chunk_outputs,
                prefix_states=prefix_states,
                q_chunks=q_chunks,
                k_chunks=k_chunks,
                gate_chunks=gate_chunks,
                beta_chunks=beta_chunks,
            )

    # Record sequential comparison timing (for performance analysis)
    seq_start = time.perf_counter()
    _sequential_prefill_fallback(
        q=q, k=k, v=v, gate=gate, beta=beta, initial_state=initial_state,
    )
    seq_end = time.perf_counter()
    recorder.increment_counter("sequential_prefill_comparison_runs")
    # Store the comparison time as metadata on a zero-duration event
    # by using a span that captures the sequential time
    with recorder.span(
        "sequential_prefill_comparison",
        mode="prefill",
        metadata={
            "sequential_time_seconds": seq_end - seq_start,
            "seq_len": seq_len,
        },
    ):
        pass  # Already timed above; span records the metadata

    return outputs, final_state


# ---------------------------------------------------------------------------
# Pipeline integration hook (Subtask 8)
# ---------------------------------------------------------------------------

# TODO(pipeline-integration): When enable_chunked_gated_deltanet_prefill is True
# in PytorchXpuOptimizationConfiguration, the pipeline_parallel_shard.py should:
#
# 1. During prefill, for each GatedDeltaNet (linear_attention) layer:
#    - Call chunked_prefill_with_fallback() instead of the sequential path
#    - Pass the chunk_size from ChunkedGatedDeltaNetPrefillConfiguration
#    - Pass the fallback_on_unsupported_shape flag from config
#    - Pass the performance_recorder for instrumentation
#
# 2. For full-attention layers: keep unchanged (they use standard HF attention)
#
# 3. Store the resulting final_state in the DynamicCache for decode continuation
#    (the GatedDeltaNetCache wrapper handles this transparently)
#
# Integration point in pipeline_parallel_shard.py._forward_layer():
#   if (
#       mode == "prefill"
#       and layer_type == "linear_attention"
#       and self._optimization_config is not None
#       and self._optimization_config.enable_chunked_gated_deltanet_prefill
#   ):
#       from exo.worker.engines.pytorch_xpu.chunked_prefill import (
#           chunked_prefill_with_fallback,
#       )
#       config = self._optimization_config.chunked_gated_deltanet_prefill
#       outputs, final_state = chunked_prefill_with_fallback(
#           q=q, k=k, v=v, gate=gate, beta=beta,
#           chunk_size=config.chunk_size,
#           fallback_on_unsupported_shape=config.fallback_on_unsupported_shape,
#           performance_recorder=self.performance_recorder,
#       )
#       # Store final_state in DynamicCache via the GatedDeltaNetCache wrapper
#       # The cache key uses the global layer index for correct decode lookup
