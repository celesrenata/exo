"""
Unit tests for OnDeviceSampler.

Tests run on CPU (mocking XPU) since we don't have XPU hardware in CI.
The class is device-agnostic — it works on any device.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import pytest
import torch

from exo.worker.engines.pytorch_xpu.on_device_sampling import OnDeviceSampler


# pin_memory=True behaves differently on MPS (Apple Silicon) — it allocates
# on MPS device instead of CPU. Skip pinned-memory tests on MPS.
_HAS_MPS = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False


class TestOnDeviceSamplerGreedy:
    """Tests for greedy (argmax) sampling on device."""

    def test_argmax_returns_highest_logit_token(self) -> None:
        """Req 8.1: argmax on device returns token ID as device tensor."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Create logits where token 42 has the highest value
        logits = torch.randn(1, 100, device=device)
        logits[0, 42] = 100.0

        token_id = sampler.sample(logits, temperature=0.0)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device
        assert token_id.item() == 42

    def test_argmax_with_near_zero_temperature(self) -> None:
        """Near-zero temperature triggers greedy decoding."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 500, device=device)
        logits[0, 123] = 50.0

        token_id = sampler.sample(logits, temperature=1e-8)

        assert token_id.item() == 123

    def test_argmax_result_stays_on_device(self) -> None:
        """Req 8.1: result tensor remains on the target device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 200, device=device)
        token_id = sampler.sample(logits, temperature=0.0)

        assert token_id.device == device


class TestOnDeviceSamplerTopK:
    """Tests for top-k sampling on device."""

    def test_top_k_returns_token_within_top_k(self) -> None:
        """Req 8.2: top-k filtering and multinomial sampling on device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Create logits where only tokens 10, 20, 30 have high values
        logits = torch.full((1, 100), -100.0, device=device)
        logits[0, 10] = 10.0
        logits[0, 20] = 9.0
        logits[0, 30] = 8.0

        torch.manual_seed(42)
        token_id = sampler.sample(logits, temperature=1.0, top_k=3)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device
        assert token_id.item() in {10, 20, 30}

    def test_top_k_with_temperature(self) -> None:
        """Top-k with temperature scaling stays on device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 500, device=device)
        torch.manual_seed(0)
        token_id = sampler.sample(logits, temperature=0.5, top_k=10)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device

    def test_top_k_larger_than_vocab_does_not_crash(self) -> None:
        """Top-k larger than vocabulary size is clamped."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 50, device=device)
        torch.manual_seed(0)
        token_id = sampler.sample(logits, temperature=1.0, top_k=1000)

        assert token_id.shape == (1,)
        assert 0 <= token_id.item() < 50


class TestOnDeviceSamplerTopP:
    """Tests for top-p (nucleus) sampling on device."""

    def test_top_p_returns_token_from_nucleus(self) -> None:
        """Req 8.3: top-p filtering and multinomial sampling on device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Create logits where one token dominates
        logits = torch.full((1, 100), -100.0, device=device)
        logits[0, 5] = 20.0  # This token has ~100% probability
        logits[0, 15] = 0.0

        torch.manual_seed(42)
        token_id = sampler.sample(logits, temperature=1.0, top_p=0.9)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device
        # With such extreme logits, token 5 dominates the nucleus
        assert token_id.item() == 5

    def test_top_p_with_uniform_logits(self) -> None:
        """Top-p with uniform logits samples from the nucleus subset."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Uniform logits — top_p=0.1 should restrict to ~10% of vocab
        logits = torch.zeros(1, 100, device=device)
        torch.manual_seed(42)
        token_id = sampler.sample(logits, temperature=1.0, top_p=0.1)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert 0 <= token_id.item() < 100

    def test_top_p_result_stays_on_device(self) -> None:
        """Req 8.3: result tensor remains on the target device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 200, device=device)
        torch.manual_seed(0)
        token_id = sampler.sample(logits, temperature=1.0, top_p=0.95)

        assert token_id.device == device


class TestOnDeviceSamplerTransfer:
    """Tests for async token ID transfer to CPU."""

    @pytest.mark.skipif(_HAS_MPS, reason="pin_memory behaves differently on MPS")
    def test_transfer_returns_cpu_tensor(self) -> None:
        """Req 8.4: transfer only single int64 token ID to CPU."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        token_id_device = torch.tensor([42], dtype=torch.long, device=device)
        cpu_tensor = sampler.transfer_token_to_cpu_async(token_id_device)

        assert cpu_tensor.device == torch.device("cpu")
        assert cpu_tensor.dtype == torch.long
        assert cpu_tensor.shape == (1,)
        assert cpu_tensor.item() == 42

    @pytest.mark.skipif(_HAS_MPS, reason="pin_memory behaves differently on MPS")
    def test_transfer_returns_pinned_memory(self) -> None:
        """Req 8.5: CPU tensor uses pinned memory for async DMA."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        token_id_device = torch.tensor([99], dtype=torch.long, device=device)
        cpu_tensor = sampler.transfer_token_to_cpu_async(token_id_device)

        # pin_memory=True on CPU allocator — verify the tensor is pinned
        assert cpu_tensor.is_pinned()

    @pytest.mark.skipif(_HAS_MPS, reason="pin_memory behaves differently on MPS")
    def test_transfer_preserves_token_value(self) -> None:
        """Transfer preserves the exact token ID value."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        for token_value in [0, 1, 1000, 151643]:  # include EOS token ID
            token_id_device = torch.tensor(
                [token_value], dtype=torch.long, device=device
            )
            cpu_tensor = sampler.transfer_token_to_cpu_async(token_id_device)
            assert cpu_tensor.item() == token_value


class TestOnDeviceSamplerCombinedTopKTopP:
    """Tests for combined top-k + top-p sampling on device."""

    def test_combined_top_k_top_p_returns_valid_token(self) -> None:
        """Combined top-k + top-p sampling stays on device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Create logits where tokens 10, 20, 30 have high values
        logits = torch.full((1, 100), -100.0, device=device)
        logits[0, 10] = 10.0
        logits[0, 20] = 9.0
        logits[0, 30] = 8.0

        torch.manual_seed(42)
        token_id = sampler.sample(logits, temperature=1.0, top_k=5, top_p=0.9)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device
        assert token_id.item() in {10, 20, 30}

    def test_combined_top_k_top_p_respects_nucleus(self) -> None:
        """Combined sampling: top-p further restricts the top-k set."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        # Token 0 dominates — with tight top_p, only token 0 should be selected
        logits = torch.full((1, 50), -100.0, device=device)
        logits[0, 0] = 50.0
        logits[0, 1] = 1.0
        logits[0, 2] = 0.5

        torch.manual_seed(0)
        token_id = sampler.sample(logits, temperature=1.0, top_k=10, top_p=0.5)

        # Token 0 has overwhelming probability, top_p=0.5 keeps only it
        assert token_id.item() == 0


class TestOnDeviceSamplerPlainMultinomial:
    """Tests for plain multinomial sampling (temperature != 0, no top-k/top-p)."""

    def test_plain_multinomial_stays_on_device(self) -> None:
        """Multinomial sampling without top-k/top-p stays on device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)

        logits = torch.randn(1, 100, device=device)
        torch.manual_seed(42)
        token_id = sampler.sample(logits, temperature=1.0)

        assert token_id.shape == (1,)
        assert token_id.dtype == torch.long
        assert token_id.device == device
        assert 0 <= token_id.item() < 100

    def test_device_property(self) -> None:
        """The device property returns the configured device."""
        device = torch.device("cpu")
        sampler = OnDeviceSampler(device=device)
        assert sampler.device == device
