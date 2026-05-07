"""Token sampling with temperature, top-k, and top-p on GPU tensors.

Implements temperature scaling, top-k filtering, and nucleus (top-p) sampling
for token generation. All operations work on GPU tensors (device-agnostic)
and return numpy arrays of sampled token IDs.

Requirements: 2.6
"""

from __future__ import annotations

import typing

import numpy as np
import torch


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> np.ndarray:
    """Sample token IDs from logits using temperature, top-k, and top-p.

    Applies the following pipeline:
      1. Temperature scaling (higher = more random)
      2. Top-k filtering (keep only top-k logits)
      3. Top-p / nucleus filtering (keep smallest set with cumulative prob >= top_p)
      4. Softmax to convert to probabilities
      5. Multinomial sampling from the distribution

    Args:
        logits: Raw logits tensor of shape (batch_size, vocab_size) or (vocab_size,).
        temperature: Temperature for scaling. 0 means greedy (argmax).
        top_k: If > 0, keep only the top-k highest logits.
        top_p: If < 1.0, use nucleus sampling (keep smallest set with
            cumulative probability >= top_p).

    Returns:
        Numpy array of sampled token IDs with shape (batch_size,) or (1,).
    """
    # Handle 1D input by adding batch dimension
    unbatched = logits.dim() == 1
    if unbatched:
        logits = logits.unsqueeze(0)

    # Greedy decoding: temperature == 0 means argmax
    if temperature == 0.0:
        token_ids = torch.argmax(logits, dim=-1)
        result = token_ids.cpu().numpy()
        return result

    # Step 1: Temperature scaling
    logits = logits / temperature

    # Step 2: Top-k filtering
    if top_k > 0:
        logits = _apply_top_k(logits, top_k)

    # Step 3: Top-p (nucleus) filtering
    if top_p < 1.0:
        logits = _apply_top_p(logits, top_p)

    # Step 4: Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Step 5: Sample from the distribution
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return token_ids.cpu().numpy()


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out logits below the top-k threshold by setting them to -inf.

    Args:
        logits: Tensor of shape (batch_size, vocab_size).
        top_k: Number of top logits to keep.

    Returns:
        Filtered logits tensor with non-top-k values set to -inf.
    """
    # Clamp top_k to vocab size
    top_k = min(top_k, logits.size(-1))

    # Get the k-th largest value as threshold
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)

    # Set values below threshold to -inf
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return logits


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering.

    Keeps the smallest set of tokens whose cumulative probability is >= top_p.
    Tokens outside this set are set to -inf.

    Args:
        logits: Tensor of shape (batch_size, vocab_size).
        top_p: Cumulative probability threshold.

    Returns:
        Filtered logits tensor with low-probability tokens set to -inf.
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities from sorted logits
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask for tokens to remove (cumulative prob exceeds top_p)
    # Shift right so the token that pushes over the threshold is kept
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p

    # Set masked logits to -inf in sorted order
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original order
    logits = torch.zeros_like(logits).scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )
    return logits
