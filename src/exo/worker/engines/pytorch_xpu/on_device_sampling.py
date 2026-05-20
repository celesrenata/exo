"""
On-device token sampling that executes entirely on XPU.

Keeps all sampling operations (argmax, top-k, top-p) on the XPU device,
avoiding the transfer of full logits tensors to CPU. The only data that
crosses the device boundary is a single int64 scalar token ID, transferred
asynchronously via pinned memory to overlap with the next decode step.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import logging
from typing import final

import torch

logger = logging.getLogger(__name__)


@final
class OnDeviceSampler:
    """Token sampling that executes entirely on XPU device.

    All sampling operations (argmax, top-k filtering, top-p nucleus filtering,
    multinomial sampling) run on the target device. No logits are transferred
    to CPU. The only CPU transfer is a single int64 token ID via non-blocking
    copy to pinned memory.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device

    @property
    def device(self) -> torch.device:
        """The device on which sampling operations execute."""
        return self._device

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Sample next token ID on device. Returns [1] int64 tensor on device.

        No CPU transfer occurs here. All operations stay on the target device.

        Args:
            logits: Tensor of shape [1, vocab_size] on the target device.
                Raw logit values from the model's final layer.
            temperature: Sampling temperature. Values <= 0 or very close to 0
                trigger greedy (argmax) decoding.
            top_k: Number of highest-probability tokens to keep. None disables
                top-k filtering.
            top_p: Cumulative probability threshold for nucleus filtering.
                None disables top-p filtering. Must be in (0, 1] when set.

        Returns:
            Tensor of shape [1] with dtype int64 on the target device,
            containing the sampled token ID.
        """
        # Flatten to [vocab_size] for processing
        if logits.ndim == 2:
            logits = logits.squeeze(0)

        # Greedy decoding: temperature <= 0 or near-zero
        if temperature <= 1e-7:
            token_id = torch.argmax(logits, dim=-1).unsqueeze(0).to(torch.long)
            return token_id

        # Temperature scaling (on device)
        if temperature != 1.0:
            logits = logits / temperature

        # Combined top-k + top-p: apply top-k first, then top-p on filtered set
        if top_k is not None and top_k > 0 and top_p is not None and 0.0 < top_p < 1.0:
            return self._sample_top_k_top_p(logits, top_k, top_p)

        # Top-k filtering only (on device)
        if top_k is not None and top_k > 0:
            return self._sample_top_k(logits, top_k)

        # Top-p (nucleus) filtering only (on device)
        if top_p is not None and 0.0 < top_p < 1.0:
            return self._sample_top_p(logits, top_p)

        # Plain multinomial sampling with temperature (no top-k/top-p)
        probabilities = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probabilities.unsqueeze(0), num_samples=1).squeeze(0).to(torch.long)
        return token_id

    def _sample_top_k(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Top-k sampling entirely on device.

        Uses torch.topk to select the k highest logits, applies softmax
        over the candidates, and samples via torch.multinomial.

        Args:
            logits: 1D tensor of shape [vocab_size], already temperature-scaled.
            top_k: Number of top candidates to consider.

        Returns:
            Tensor of shape [1] with dtype int64 on the target device.
        """
        vocabulary_size = logits.size(0)
        effective_k = min(top_k, vocabulary_size)

        # Extract top-k values and indices (on device)
        top_k_values, top_k_indices = torch.topk(logits, effective_k)

        # Convert to probabilities (on device)
        top_k_probabilities = torch.softmax(top_k_values, dim=-1)

        # Sample from top-k distribution (on device)
        try:
            sampled_position = torch.multinomial(
                top_k_probabilities.unsqueeze(0), num_samples=1
            ).squeeze(0)
            token_id = top_k_indices[sampled_position].to(torch.long)
        except RuntimeError:
            # Fallback to argmax within top-k on multinomial failure
            logger.warning(
                "On-device multinomial failed during top-k sampling, "
                "falling back to argmax"
            )
            token_id = top_k_indices[0:1].to(torch.long)

        return token_id

    def _sample_top_p(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling entirely on device.

        Sorts logits descending, computes cumulative probabilities, masks
        tokens beyond the nucleus threshold, and samples via torch.multinomial.

        Args:
            logits: 1D tensor of shape [vocab_size], already temperature-scaled.
            top_p: Cumulative probability threshold in (0, 1).

        Returns:
            Tensor of shape [1] with dtype int64 on the target device.
        """
        # Sort logits descending (on device)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities from sorted logits (on device)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # Create mask: remove tokens with cumulative probability above threshold
        # Keep the first token that crosses the threshold (shift mask by 1)
        sorted_mask = torch.zeros_like(cumulative_probabilities, dtype=torch.bool)
        sorted_mask[1:] = cumulative_probabilities[:-1] >= top_p

        # Set masked logits to -inf (on device)
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # Convert filtered logits to probabilities (on device)
        filtered_probabilities = torch.softmax(sorted_logits, dim=-1)

        # Sample from the nucleus distribution (on device)
        try:
            sampled_position = torch.multinomial(
                filtered_probabilities.unsqueeze(0), num_samples=1
            ).squeeze(0)
            token_id = sorted_indices[sampled_position].to(torch.long)
        except RuntimeError:
            # Fallback to the highest-probability token on multinomial failure
            logger.warning(
                "On-device multinomial failed during top-p sampling, "
                "falling back to argmax"
            )
            token_id = sorted_indices[0:1].to(torch.long)

        return token_id

    def _sample_top_k_top_p(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Combined top-k + top-p sampling entirely on device.

        Applies top-k first to reduce the candidate set, then applies top-p
        nucleus filtering on the remaining candidates before sampling.

        Args:
            logits: 1D tensor of shape [vocab_size], already temperature-scaled.
            top_k: Number of top candidates to consider.
            top_p: Cumulative probability threshold in (0, 1).

        Returns:
            Tensor of shape [1] with dtype int64 on the target device.
        """
        vocabulary_size = logits.size(0)
        effective_k = min(top_k, vocabulary_size)

        # Step 1: Top-k filtering (on device)
        top_k_values, top_k_indices = torch.topk(logits, effective_k)

        # Step 2: Sort the top-k candidates descending (already sorted by topk)
        # torch.topk returns values in descending order, so we can use them directly
        sorted_probabilities = torch.softmax(top_k_values, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # Step 3: Apply top-p mask on the top-k candidates
        sorted_mask = torch.zeros_like(cumulative_probabilities, dtype=torch.bool)
        sorted_mask[1:] = cumulative_probabilities[:-1] >= top_p

        # Mask out tokens beyond the nucleus threshold
        filtered_values = top_k_values.masked_fill(sorted_mask, float("-inf"))

        # Convert to probabilities and sample (on device)
        filtered_probabilities = torch.softmax(filtered_values, dim=-1)

        try:
            sampled_position = torch.multinomial(
                filtered_probabilities.unsqueeze(0), num_samples=1
            ).squeeze(0)
            token_id = top_k_indices[sampled_position].to(torch.long)
        except RuntimeError:
            logger.warning(
                "On-device multinomial failed during top-k+top-p sampling, "
                "falling back to argmax"
            )
            token_id = top_k_indices[0:1].to(torch.long)

        return token_id

    def transfer_token_to_cpu_async(
        self,
        token_id_xpu: torch.Tensor,
    ) -> torch.Tensor:
        """Non-blocking copy of scalar token ID to CPU pinned memory.

        Returns a CPU tensor that will be populated when the device stream
        completes. The caller can launch the next forward pass before reading
        this value — the DMA transfer overlaps with compute.

        Args:
            token_id_xpu: Tensor of shape [1] with dtype int64 on the target
                device, containing the sampled token ID.

        Returns:
            CPU tensor of shape [1] with dtype int64 in pinned memory.
            The value is valid after the device stream synchronizes.
        """
        cpu_tensor = torch.empty(1, dtype=torch.long, device="cpu", pin_memory=True)
        cpu_tensor.copy_(token_id_xpu, non_blocking=True)
        return cpu_tensor
