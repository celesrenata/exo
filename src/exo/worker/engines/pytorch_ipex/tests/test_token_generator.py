"""
Tests for TokenGenerator

These tests validate token sampling functionality including temperature,
top-k, top-p sampling, and special token handling.
"""

import pytest


# Skip all tests if PyTorch is not available
pytest.importorskip("torch")

import torch

from exo.worker.engines.pytorch_ipex.errors import InferenceError
from exo.worker.engines.pytorch_ipex.token_generator import SamplingResult, TokenGenerator


def test_token_generator_initialization() -> None:
    """Test that TokenGenerator initializes correctly."""
    generator = TokenGenerator()
    assert generator is not None


def test_token_generator_with_special_tokens() -> None:
    """Test TokenGenerator initialization with special tokens."""
    generator = TokenGenerator(eos_token_id=2, pad_token_id=0)
    assert generator is not None


def test_basic_sampling() -> None:
    """Test basic token sampling without filtering."""
    generator = TokenGenerator()
    vocab_size = 1000
    logits = torch.randn(vocab_size)

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size
    assert 0.0 <= result.probability <= 1.0
    assert result.is_eos is False
    assert result.is_pad is False


def test_temperature_sampling() -> None:
    """Test temperature scaling."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.randn(vocab_size)

    # Low temperature should produce more deterministic results
    result_low = generator.sample(logits, temperature=0.1, top_p=1.0, top_k=0)
    assert isinstance(result_low, SamplingResult)

    # High temperature should produce more random results
    result_high = generator.sample(logits, temperature=2.0, top_p=1.0, top_k=0)
    assert isinstance(result_high, SamplingResult)


def test_top_k_sampling() -> None:
    """Test top-k filtering."""
    generator = TokenGenerator()
    vocab_size = 1000
    logits = torch.randn(vocab_size)

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=50)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_top_p_sampling() -> None:
    """Test top-p (nucleus) filtering."""
    generator = TokenGenerator()
    vocab_size = 1000
    logits = torch.randn(vocab_size)

    result = generator.sample(logits, temperature=1.0, top_p=0.9, top_k=0)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_combined_sampling() -> None:
    """Test combined temperature, top-k, and top-p sampling."""
    generator = TokenGenerator()
    vocab_size = 1000
    logits = torch.randn(vocab_size)

    result = generator.sample(logits, temperature=0.8, top_p=0.95, top_k=100)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_eos_token_detection() -> None:
    """Test EOS token detection."""
    eos_token_id = 2
    generator = TokenGenerator(eos_token_id=eos_token_id)

    vocab_size = 100
    logits = torch.zeros(vocab_size)
    logits[eos_token_id] = 100.0  # Make EOS token very likely

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert result.token_id == eos_token_id
    assert result.is_eos is True
    assert result.is_pad is False


def test_pad_token_detection() -> None:
    """Test PAD token detection."""
    pad_token_id = 0
    generator = TokenGenerator(pad_token_id=pad_token_id)

    vocab_size = 100
    logits = torch.zeros(vocab_size)
    logits[pad_token_id] = 100.0  # Make PAD token very likely

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert result.token_id == pad_token_id
    assert result.is_eos is False
    assert result.is_pad is True


def test_set_special_tokens() -> None:
    """Test updating special tokens after initialization."""
    generator = TokenGenerator()

    # Initially no special tokens
    vocab_size = 100
    logits = torch.zeros(vocab_size)
    logits[5] = 100.0

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)
    assert result.is_eos is False

    # Update special tokens
    generator.set_special_tokens(eos_token_id=5, pad_token_id=1)

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)
    assert result.is_eos is True


def test_2d_logits() -> None:
    """Test sampling from 2D logits (batch dimension)."""
    generator = TokenGenerator()
    vocab_size = 1000
    batch_size = 2
    logits = torch.randn(batch_size, vocab_size)

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_invalid_temperature() -> None:
    """Test that invalid temperature raises error."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.randn(vocab_size)

    with pytest.raises(InferenceError):
        generator.sample(logits, temperature=0.0, top_p=1.0, top_k=0)

    with pytest.raises(InferenceError):
        generator.sample(logits, temperature=-1.0, top_p=1.0, top_k=0)


def test_nan_logits() -> None:
    """Test that NaN logits raise error."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.full((vocab_size,), float("nan"))

    with pytest.raises(InferenceError):
        generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)


def test_all_inf_logits() -> None:
    """Test that all infinite logits raise error."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.full((vocab_size,), float("inf"))

    with pytest.raises(InferenceError):
        generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)


def test_invalid_logits_shape() -> None:
    """Test that invalid logits shape raises error."""
    generator = TokenGenerator()
    logits = torch.randn(2, 3, 100)  # 3D tensor

    with pytest.raises(InferenceError):
        generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)


def test_top_k_larger_than_vocab() -> None:
    """Test top-k filtering when k is larger than vocab size."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.randn(vocab_size)

    # top_k larger than vocab_size should not filter anything
    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=200)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_top_p_one() -> None:
    """Test that top_p=1.0 does not filter."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.randn(vocab_size)

    result = generator.sample(logits, temperature=1.0, top_p=1.0, top_k=0)

    assert isinstance(result, SamplingResult)
    assert 0 <= result.token_id < vocab_size


def test_sampling_result_immutable() -> None:
    """Test that SamplingResult is immutable (frozen dataclass)."""
    result = SamplingResult(token_id=42, is_eos=False, is_pad=False, probability=0.5)

    # Should not be able to modify frozen dataclass
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        result.token_id = 100  # type: ignore


def test_deterministic_sampling() -> None:
    """Test that sampling with very low temperature is deterministic."""
    generator = TokenGenerator()
    vocab_size = 100
    logits = torch.randn(vocab_size)

    # Sample multiple times with very low temperature
    results = [
        generator.sample(logits, temperature=0.01, top_p=1.0, top_k=0)
        for _ in range(5)
    ]

    # All results should be the same token (highest logit)
    token_ids = [r.token_id for r in results]
    assert len(set(token_ids)) == 1, "Low temperature sampling should be deterministic"
