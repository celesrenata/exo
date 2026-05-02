"""
Token Generator for PyTorch XPU Backend

This module provides token sampling functionality for the PyTorch XPU
inference backend, implementing temperature scaling, top-k, and top-p sampling.

Requirements addressed:
- 3.3: Token sampling with temperature, top-p, and top-k
- 6.3: Special token handling (EOS, PAD)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, final

from exo.worker.engines.pytorch_xpu.errors import InferenceError

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingResult:
    """
    Result of token sampling operation.

    Attributes:
        token_id: The sampled token ID
        is_eos: Whether this is an end-of-sequence token
        is_pad: Whether this is a padding token
        probability: Probability of the sampled token
    """

    token_id: int
    is_eos: bool
    is_pad: bool
    probability: float


@final
class TokenGenerator:
    """
    Token generator for sampling from model logits.

    This class implements various sampling strategies:
    - Temperature scaling
    - Top-k filtering
    - Top-p (nucleus) sampling
    - Special token detection (EOS, PAD)

    Requirements: 3.3, 6.3
    """

    def __init__(
        self,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """
        Initialize TokenGenerator.

        Args:
            eos_token_id: End-of-sequence token ID (if known)
            pad_token_id: Padding token ID (if known)

        Requirements: 3.3
        """
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

        # Try to import torch
        try:
            import torch  # type: ignore

            self._torch = torch
            logger.debug("TokenGenerator initialized with PyTorch")
        except ImportError:
            logger.error("PyTorch not available - TokenGenerator cannot function")
            raise RuntimeError("PyTorch is required for TokenGenerator")

    def sample(
        self,
        logits: "torch.Tensor",  # type: ignore
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> SamplingResult:
        """
        Sample a token from logits using specified sampling strategy.

        This method:
        1. Applies temperature scaling
        2. Applies top-k filtering (if top_k > 0)
        3. Applies top-p (nucleus) filtering (if top_p < 1.0)
        4. Samples from the filtered distribution
        5. Detects special tokens (EOS, PAD)

        Args:
            logits: Logit tensor (shape: [vocab_size] or [batch_size, vocab_size])
            temperature: Temperature for scaling logits (higher = more random)
            top_p: Cumulative probability threshold for nucleus sampling
            top_k: Number of top tokens to consider (0 = disabled)

        Returns:
            SamplingResult with token ID and metadata

        Raises:
            InferenceError: If sampling fails or produces invalid results

        Requirements: 3.3, 6.3
        """
        try:
            # Ensure logits is 1D or 2D
            if len(logits.shape) > 2:
                raise InferenceError(
                    message=f"Invalid logits shape: {logits.shape}, expected 1D or 2D"
                )

            # If 2D, take the last batch element
            if len(logits.shape) == 2:
                logits = logits[-1, :]  # [vocab_size]

            # Check for NaN or invalid logits
            if self._torch.isnan(logits).any():
                raise InferenceError(message="Logits contain NaN values")

            if self._torch.isinf(logits).all():
                raise InferenceError(message="All logits are infinite")

            # Apply temperature scaling
            if temperature != 1.0:
                if temperature <= 0:
                    raise InferenceError(
                        message=f"Invalid temperature: {temperature}, must be > 0"
                    )
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                logits = self._apply_top_k(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = self._apply_top_p(logits, top_p)

            # Convert to probabilities
            probs = self._torch.nn.functional.softmax(logits, dim=-1)

            # Check for valid probability distribution
            if self._torch.isnan(probs).any():
                raise InferenceError(
                    message="Invalid probability distribution after softmax"
                )

            if (probs < 0).any():
                raise InferenceError(
                    message="Negative probabilities after softmax"
                )

            # Sample from distribution
            sampled_token_tensor = self._torch.multinomial(probs, num_samples=1)
            token_id = int(sampled_token_tensor.item())

            # Get probability of sampled token
            token_prob = float(probs[token_id].item())

            # Check for special tokens
            is_eos = self._is_eos_token(token_id)
            is_pad = self._is_pad_token(token_id)

            logger.debug(
                f"Sampled token {token_id} with probability {token_prob:.4f} "
                f"(EOS: {is_eos}, PAD: {is_pad})"
            )

            return SamplingResult(
                token_id=token_id,
                is_eos=is_eos,
                is_pad=is_pad,
                probability=token_prob,
            )

        except InferenceError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise InferenceError(
                message="Unexpected sampling error", original_error=e
            ) from e

    def _apply_top_k(
        self, logits: "torch.Tensor", top_k: int  # type: ignore
    ) -> "torch.Tensor":  # type: ignore
        """
        Apply top-k filtering to logits.

        Keeps only the top-k highest logits and sets others to -inf.

        Args:
            logits: Logit tensor (shape: [vocab_size])
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits tensor

        Requirements: 3.3
        """
        if top_k >= logits.size(-1):
            # No filtering needed
            return logits

        # Get top-k values and indices
        top_k_values, _ = self._torch.topk(logits, top_k)
        min_top_k_value = top_k_values[-1]

        # Set logits below threshold to -inf
        indices_to_remove = logits < min_top_k_value
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

        return logits

    def _apply_top_p(
        self, logits: "torch.Tensor", top_p: float  # type: ignore
    ) -> "torch.Tensor":  # type: ignore
        """
        Apply top-p (nucleus) filtering to logits.

        Keeps only tokens whose cumulative probability is below top_p.

        Args:
            logits: Logit tensor (shape: [vocab_size])
            top_p: Cumulative probability threshold

        Returns:
            Filtered logits tensor

        Requirements: 3.3
        """
        if top_p >= 1.0:
            # No filtering needed
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = self._torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        cumulative_probs = self._torch.cumsum(
            self._torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # Scatter back to original indexing
        indices_to_remove = self._torch.zeros_like(logits, dtype=self._torch.bool)
        indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True

        # Set removed indices to -inf
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

        return logits

    def _is_eos_token(self, token_id: int) -> bool:
        """
        Check if token is an end-of-sequence token.

        Args:
            token_id: Token ID to check

        Returns:
            True if token is EOS, False otherwise

        Requirements: 6.3
        """
        if self._eos_token_id is None:
            return False
        return token_id == self._eos_token_id

    def _is_pad_token(self, token_id: int) -> bool:
        """
        Check if token is a padding token.

        Args:
            token_id: Token ID to check

        Returns:
            True if token is PAD, False otherwise

        Requirements: 6.3
        """
        if self._pad_token_id is None:
            return False
        return token_id == self._pad_token_id

    def set_special_tokens(
        self, eos_token_id: Optional[int], pad_token_id: Optional[int]
    ) -> None:
        """
        Update special token IDs.

        This method allows updating special token IDs after initialization,
        which is useful when loading models with different tokenizers.

        Args:
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID

        Requirements: 6.3
        """
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        logger.debug(f"Updated special tokens: EOS={eos_token_id}, PAD={pad_token_id}")
