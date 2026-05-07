"""Unit tests for token sampling with temperature, top-k, and top-p."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exo.worker.engines.pytorch.sampling import sample_token


class TestSampleTokenGreedy:
    """Tests for greedy (temperature=0) sampling."""

    def test_greedy_selects_argmax(self) -> None:
        """Temperature=0 should always select the highest logit."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        result = sample_token(logits, temperature=0.0)
        assert result.item() == 1  # index of 5.0

    def test_greedy_batched(self) -> None:
        """Temperature=0 with batched input selects argmax per row."""
        logits = torch.tensor([
            [1.0, 5.0, 3.0],
            [7.0, 2.0, 4.0],
        ])
        result = sample_token(logits, temperature=0.0)
        np.testing.assert_array_equal(result, [1, 0])

    def test_greedy_returns_numpy_array(self) -> None:
        """Result should be a numpy array."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = sample_token(logits, temperature=0.0)
        assert isinstance(result, np.ndarray)


class TestSampleTokenTemperature:
    """Tests for temperature scaling."""

    def test_low_temperature_concentrates_probability(self) -> None:
        """Low temperature should make sampling nearly deterministic."""
        logits = torch.tensor([0.0, 10.0, 0.0, 0.0])
        # With very low temperature, the dominant logit should always win
        results = [sample_token(logits, temperature=0.01).item() for _ in range(20)]
        assert all(r == 1 for r in results)

    def test_temperature_one_is_default(self) -> None:
        """Temperature=1.0 should not modify the logits distribution."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Just verify it runs without error and returns valid indices
        result = sample_token(logits, temperature=1.0)
        assert 0 <= result.item() < 4


class TestSampleTokenTopK:
    """Tests for top-k filtering."""

    def test_top_k_restricts_to_k_tokens(self) -> None:
        """With top_k=1, only the highest logit should be sampled."""
        logits = torch.tensor([1.0, 10.0, 2.0, 3.0])
        results = [sample_token(logits, top_k=1).item() for _ in range(20)]
        assert all(r == 1 for r in results)

    def test_top_k_zero_means_no_filtering(self) -> None:
        """top_k=0 should not filter any tokens."""
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        # All tokens should be possible
        results = set(sample_token(logits, top_k=0).item() for _ in range(100))
        # With uniform logits and 100 samples, we should see multiple tokens
        assert len(results) > 1

    def test_top_k_larger_than_vocab_is_safe(self) -> None:
        """top_k larger than vocab size should not crash."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = sample_token(logits, top_k=100)
        assert 0 <= result.item() < 3


class TestSampleTokenTopP:
    """Tests for top-p (nucleus) filtering."""

    def test_top_p_one_means_no_filtering(self) -> None:
        """top_p=1.0 should not filter any tokens."""
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        results = set(sample_token(logits, top_p=1.0).item() for _ in range(100))
        assert len(results) > 1

    def test_top_p_very_small_selects_top_token(self) -> None:
        """Very small top_p should select only the most probable token."""
        logits = torch.tensor([0.0, 10.0, 0.0, 0.0])
        results = [sample_token(logits, top_p=0.01).item() for _ in range(20)]
        assert all(r == 1 for r in results)

    def test_top_p_filters_low_probability_tokens(self) -> None:
        """top_p should exclude tokens with very low probability."""
        # One dominant token and three very low ones
        logits = torch.tensor([10.0, -10.0, -10.0, -10.0])
        results = [sample_token(logits, top_p=0.9).item() for _ in range(50)]
        # The dominant token should always be selected
        assert all(r == 0 for r in results)


class TestSampleTokenCombined:
    """Tests for combined sampling parameters."""

    def test_top_k_and_top_p_together(self) -> None:
        """Both top_k and top_p can be applied simultaneously."""
        logits = torch.tensor([10.0, 9.0, -5.0, -5.0])
        # top_k=2 keeps only first two, top_p=0.9 further filters
        results = set(sample_token(logits, top_k=2, top_p=0.9).item() for _ in range(50))
        # Only indices 0 and 1 should be possible
        assert results.issubset({0, 1})

    def test_all_parameters_together(self) -> None:
        """Temperature, top_k, and top_p can all be used together."""
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        result = sample_token(logits, temperature=0.5, top_k=3, top_p=0.9)
        # Should only sample from top-3 tokens
        assert result.item() in {0, 1, 2}


class TestSampleTokenShape:
    """Tests for input/output shape handling."""

    def test_1d_input_returns_1d_output(self) -> None:
        """1D logits (vocab_size,) should return shape (1,) or scalar-like."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = sample_token(logits, temperature=0.0)
        assert result.ndim == 1

    def test_2d_input_returns_1d_output(self) -> None:
        """2D logits (batch, vocab) should return shape (batch,)."""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = sample_token(logits, temperature=0.0)
        assert result.shape == (2,)

    def test_output_dtype_is_integer(self) -> None:
        """Sampled token IDs should be integers."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = sample_token(logits, temperature=1.0)
        assert np.issubdtype(result.dtype, np.integer)
