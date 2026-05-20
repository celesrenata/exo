"""Packed projection utilities for kernel launch reduction.

Provides weight packing functions applied at model load time to reduce
kernel launches per transformer layer from 5 matmuls (Q, K, V, gate, up)
to 3 (packed_QKV, packed_gate_up, down_proj). Over 32 layers, this
eliminates 64 kernel launches per token.

Packing happens once during model loading (not during inference).
Unpacking (split) happens during inference and is torch.compile-traceable.

Qwen3.5-4B dimensions:
    - hidden_size = 2560
    - num_heads = 32, head_dim = 80
    - num_kv_heads = 4 (GQA with 8:1 ratio)
    - Q projection: [2560, 2560]
    - K projection: [320, 2560]
    - V projection: [320, 2560]
    - Packed QKV: [3200, 2560]
    - intermediate_size = 9728
    - Gate projection: [9728, 2560]
    - Up projection: [9728, 2560]
    - Packed gate/up: [19456, 2560]

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**
"""

from __future__ import annotations

import torch


def pack_qkv_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Pack Q, K, V projection weights into a single concatenated matrix.

    Concatenates along the output dimension (dim=0) so that a single
    ``torch.matmul(input, packed_weight.T)`` produces the concatenated
    Q, K, V outputs which can then be split.

    Args:
        q_weight: Q projection weight, shape ``[q_dim, hidden_size]``.
        k_weight: K projection weight, shape ``[k_dim, hidden_size]``.
        v_weight: V projection weight, shape ``[v_dim, hidden_size]``.

    Returns:
        Packed weight tensor of shape ``[q_dim + k_dim + v_dim, hidden_size]``.

    Raises:
        ValueError: If weight tensors have incompatible hidden dimensions.

    **Validates: Requirement 5.1**
    """
    if q_weight.shape[1] != k_weight.shape[1] or q_weight.shape[1] != v_weight.shape[1]:
        raise ValueError(
            f"All QKV weights must have the same hidden_size dimension. "
            f"Got Q: {q_weight.shape[1]}, K: {k_weight.shape[1]}, V: {v_weight.shape[1]}"
        )
    return torch.cat([q_weight, k_weight, v_weight], dim=0)


def unpack_qkv_output(
    packed_output: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKV projection output into separate Q, K, V tensors.

    This function is torch.compile-traceable — it uses ``torch.split()``
    which the Inductor backend handles without graph breaks.

    Args:
        packed_output: Result of matmul with packed QKV weight,
            shape ``[..., q_dim + k_dim + v_dim]``.
        q_dim: Size of the Q output dimension.
        k_dim: Size of the K output dimension.
        v_dim: Size of the V output dimension.

    Returns:
        Tuple of (Q, K, V) tensors split along the last dimension.

    **Validates: Requirement 5.2**
    """
    parts = torch.split(packed_output, [q_dim, k_dim, v_dim], dim=-1)
    return parts[0], parts[1], parts[2]


def pack_gate_up_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """Pack gate and up projection weights into a single concatenated matrix.

    Concatenates along the output dimension (dim=0) so that a single
    ``torch.matmul(input, packed_weight.T)`` produces the concatenated
    gate and up outputs which can then be split.

    Args:
        gate_weight: Gate projection weight, shape ``[intermediate_size, hidden_size]``.
        up_weight: Up projection weight, shape ``[intermediate_size, hidden_size]``.

    Returns:
        Packed weight tensor of shape ``[2 * intermediate_size, hidden_size]``.

    Raises:
        ValueError: If weight tensors have incompatible shapes.

    **Validates: Requirement 5.3**
    """
    if gate_weight.shape != up_weight.shape:
        raise ValueError(
            f"Gate and up weights must have the same shape. "
            f"Got gate: {gate_weight.shape}, up: {up_weight.shape}"
        )
    return torch.cat([gate_weight, up_weight], dim=0)


def unpack_gate_up_output(
    packed_output: torch.Tensor,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed gate/up projection output into separate gate and up tensors.

    This function is torch.compile-traceable — it uses ``torch.split()``
    which the Inductor backend handles without graph breaks.

    Args:
        packed_output: Result of matmul with packed gate/up weight,
            shape ``[..., 2 * intermediate_size]``.
        intermediate_size: Size of each projection output dimension.

    Returns:
        Tuple of (gate, up) tensors split along the last dimension.

    **Validates: Requirement 5.4**
    """
    parts = torch.split(packed_output, [intermediate_size, intermediate_size], dim=-1)
    return parts[0], parts[1]


def pack_layer_qkv_from_model(
    layer: torch.nn.Module,
) -> torch.Tensor | None:
    """Detect and pack Q, K, V projection weights from a transformer layer.

    Inspects the layer for separate ``q_proj``, ``k_proj``, ``v_proj``
    submodules (as found in HuggingFace Qwen3.5 layers) and packs their
    weight tensors into a single concatenated matrix.

    Args:
        layer: A transformer decoder layer (e.g., ``Qwen3_5DecoderLayer``).

    Returns:
        Packed QKV weight tensor, or None if the layer does not have
        separate Q/K/V projection submodules.

    **Validates: Requirement 5.1**
    """
    # Navigate to the self_attn submodule
    self_attn: torch.nn.Module | None = None
    if hasattr(layer, "self_attn"):
        candidate = layer.self_attn
        if isinstance(candidate, torch.nn.Module):
            self_attn = candidate

    if self_attn is None:
        return None

    # Extract Q, K, V projection submodules
    q_proj: torch.nn.Linear | None = None
    k_proj: torch.nn.Linear | None = None
    v_proj: torch.nn.Linear | None = None

    if hasattr(self_attn, "q_proj"):
        candidate_q = self_attn.q_proj
        if isinstance(candidate_q, torch.nn.Linear):
            q_proj = candidate_q
    if hasattr(self_attn, "k_proj"):
        candidate_k = self_attn.k_proj
        if isinstance(candidate_k, torch.nn.Linear):
            k_proj = candidate_k
    if hasattr(self_attn, "v_proj"):
        candidate_v = self_attn.v_proj
        if isinstance(candidate_v, torch.nn.Linear):
            v_proj = candidate_v

    if q_proj is None or k_proj is None or v_proj is None:
        return None

    return pack_qkv_weights(q_proj.weight.data, k_proj.weight.data, v_proj.weight.data)


def pack_layer_gate_up_from_model(
    layer: torch.nn.Module,
) -> torch.Tensor | None:
    """Detect and pack gate and up projection weights from a transformer layer.

    Inspects the layer for separate ``gate_proj`` and ``up_proj``
    submodules (as found in HuggingFace Qwen3.5 MLP layers) and packs
    their weight tensors into a single concatenated matrix.

    Args:
        layer: A transformer decoder layer (e.g., ``Qwen3_5DecoderLayer``).

    Returns:
        Packed gate/up weight tensor, or None if the layer does not have
        separate gate/up projection submodules.

    **Validates: Requirement 5.3**
    """
    # Navigate to the mlp submodule
    mlp: torch.nn.Module | None = None
    if hasattr(layer, "mlp"):
        candidate = layer.mlp
        if isinstance(candidate, torch.nn.Module):
            mlp = candidate

    if mlp is None:
        return None

    # Extract gate and up projection submodules
    gate_proj: torch.nn.Linear | None = None
    up_proj: torch.nn.Linear | None = None

    if hasattr(mlp, "gate_proj"):
        candidate_gate = mlp.gate_proj
        if isinstance(candidate_gate, torch.nn.Linear):
            gate_proj = candidate_gate
    if hasattr(mlp, "up_proj"):
        candidate_up = mlp.up_proj
        if isinstance(candidate_up, torch.nn.Linear):
            up_proj = candidate_up

    if gate_proj is None or up_proj is None:
        return None

    return pack_gate_up_weights(gate_proj.weight.data, up_proj.weight.data)
