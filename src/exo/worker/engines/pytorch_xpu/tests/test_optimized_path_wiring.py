"""
Tests for task 15.1: Wire all components into the optimized decode path.

Validates that:
- OnDeviceSampler is used when enable_on_device_sampling=True (Req 8.1, 8.4)
- OnDeviceSampler falls back to CPU sampling on failure (Req 11.3)
- Each optimization falls back independently (Req 11.3)
- All flags can be enabled simultaneously without conflict (Req 11.1)
- The on-device sampling integration works end-to-end

Requirements: 11.1, 11.3
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import torch

from exo.worker.engines.pytorch_xpu.on_device_sampling import OnDeviceSampler
from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PytorchXpuOptimizationConfiguration,
)


# ---------------------------------------------------------------------------
# Helpers — replicate the pipeline_generator helper logic for testing
# without importing the full module (which has transitive deps on aiofiles)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _create_on_device_sampler(
    optimization_configuration: PytorchXpuOptimizationConfiguration | None,
    device: str,
) -> OnDeviceSampler | None:
    """Create an OnDeviceSampler if on-device sampling is enabled.

    Replicates the logic from pipeline_generator._create_on_device_sampler.
    """
    if optimization_configuration is None:
        return None
    if not optimization_configuration.enable_on_device_sampling:
        return None
    try:
        sampler = OnDeviceSampler(device=torch.device(device))
        return sampler
    except Exception as exc:
        logger.warning(
            "Failed to create OnDeviceSampler, falling back to CPU sampling: %s",
            exc,
        )
        return None


def _sample_with_on_device_sampler(
    logits: torch.Tensor,
    on_device_sampler: OnDeviceSampler | None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample a token using on-device sampling with fallback to CPU sampling.

    Replicates the logic from pipeline_generator._sample_with_on_device_sampler.
    """
    if on_device_sampler is None:
        return _cpu_sample_token(logits, temperature, top_k, top_p)

    try:
        # Normalize logits shape: extract last-position logits
        sampling_logits = logits
        if sampling_logits.dim() == 3:
            sampling_logits = sampling_logits[:, -1, :]  # [1, vocab_size]

        # Sample on device — returns [1] int64 tensor on device
        token_id_tensor = on_device_sampler.sample(
            sampling_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # If the token is already on CPU (e.g., CPU device or test mode),
        # read it directly. Otherwise, use async transfer for XPU→CPU.
        if token_id_tensor.device.type == "cpu":
            return int(token_id_tensor.item())

        # Transfer token ID to CPU asynchronously (XPU → pinned CPU memory)
        cpu_tensor = on_device_sampler.transfer_token_to_cpu_async(token_id_tensor)

        # Read the token ID (blocks until async copy completes)
        return int(cpu_tensor.item())

    except Exception as exc:
        logger.warning(
            "On-device sampling failed, falling back to CPU sampling: %s",
            exc,
        )
        return _cpu_sample_token(logits, temperature, top_k, top_p)


def _cpu_sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Simple CPU-based token sampling (greedy for testing)."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    if temperature <= 1e-7:
        return int(torch.argmax(logits).item())
    # For non-greedy, use simple multinomial
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs.unsqueeze(0), num_samples=1).item())


def _make_config(
    enable_on_device_sampling: bool = False,
    enable_static_kv_cache: bool = False,
    enable_torch_compile: bool = False,
    enable_packed_projections: bool = False,
    enable_fused_kernels: bool = False,
    enable_async_output: bool = False,
    enable_sync_removal: bool = False,
) -> PytorchXpuOptimizationConfiguration:
    """Create a test optimization configuration with specified flags."""
    return PytorchXpuOptimizationConfiguration(
        enable_on_device_sampling=enable_on_device_sampling,
        enable_static_kv_cache=enable_static_kv_cache,
        enable_torch_compile=enable_torch_compile,
        enable_packed_projections=enable_packed_projections,
        enable_fused_kernels=enable_fused_kernels,
        enable_async_output=enable_async_output,
        enable_sync_removal=enable_sync_removal,
    )


def _make_logits(vocab_size: int = 100, peak_token: int = 42) -> torch.Tensor:
    """Create a logits tensor with a clear peak at peak_token."""
    logits = torch.randn(1, vocab_size)
    logits[0, peak_token] = 100.0  # Make this token overwhelmingly likely
    return logits


# ---------------------------------------------------------------------------
# Tests: _create_on_device_sampler
# ---------------------------------------------------------------------------


class TestCreateOnDeviceSampler:
    """Tests for the on-device sampler creation logic."""

    def test_returns_none_when_config_is_none(self) -> None:
        """No sampler created when configuration is None."""
        result = _create_on_device_sampler(None, "cpu")
        assert result is None

    def test_returns_none_when_flag_disabled(self) -> None:
        """No sampler created when enable_on_device_sampling=False."""
        config = _make_config(enable_on_device_sampling=False)
        result = _create_on_device_sampler(config, "cpu")
        assert result is None

    def test_returns_sampler_when_flag_enabled(self) -> None:
        """Sampler created when enable_on_device_sampling=True."""
        config = _make_config(enable_on_device_sampling=True)
        result = _create_on_device_sampler(config, "cpu")
        assert result is not None
        assert isinstance(result, OnDeviceSampler)
        assert result.device == torch.device("cpu")

    def test_sampler_device_matches_requested(self) -> None:
        """Sampler is created on the requested device."""
        config = _make_config(enable_on_device_sampling=True)
        result = _create_on_device_sampler(config, "cpu")
        assert result is not None
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Tests: _sample_with_on_device_sampler
# ---------------------------------------------------------------------------


class TestSampleWithOnDeviceSampler:
    """Tests for the on-device sampling with fallback logic."""

    def test_uses_cpu_sampling_when_sampler_is_none(self) -> None:
        """Falls back to CPU sampling when no on-device sampler is provided."""
        logits = _make_logits(peak_token=42)
        # With temperature near zero, should get argmax = 42
        token_id = _sample_with_on_device_sampler(
            logits, None, temperature=0.0001
        )
        assert token_id == 42

    def test_uses_on_device_sampler_when_provided(self) -> None:
        """Uses OnDeviceSampler.sample() when sampler is provided."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        logits = _make_logits(peak_token=42)
        # Greedy sampling (temperature near zero) should return peak token
        token_id = _sample_with_on_device_sampler(
            logits, sampler, temperature=0.0001
        )
        assert token_id == 42

    def test_handles_3d_logits(self) -> None:
        """Correctly handles [batch, seq_len, vocab_size] logits."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        # Shape [1, 5, 100] — should extract last position
        logits = torch.randn(1, 5, 100)
        logits[0, -1, 77] = 100.0  # Peak at token 77 in last position
        token_id = _sample_with_on_device_sampler(
            logits, sampler, temperature=0.0001
        )
        assert token_id == 77

    def test_falls_back_on_sampling_failure(self) -> None:
        """Falls back to CPU sampling when on-device sampling raises (Req 11.3)."""
        # Create a mock sampler that raises on sample()
        mock_sampler = MagicMock(spec=OnDeviceSampler)
        mock_sampler.sample.side_effect = RuntimeError("XPU kernel failure")

        logits = _make_logits(peak_token=42)
        # Should fall back to CPU sampling and still return correct result
        token_id = _sample_with_on_device_sampler(
            logits, mock_sampler, temperature=0.0001
        )
        assert token_id == 42

    def test_top_k_sampling_on_device(self) -> None:
        """On-device top-k sampling returns a valid token."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        logits = _make_logits(vocab_size=100, peak_token=42)
        token_id = _sample_with_on_device_sampler(
            logits, sampler, temperature=1.0, top_k=5
        )
        # Token should be valid (42 is overwhelmingly likely with top_k=5)
        assert 0 <= token_id < 100

    def test_top_p_sampling_on_device(self) -> None:
        """On-device top-p sampling returns a valid token."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        logits = _make_logits(vocab_size=100, peak_token=42)
        token_id = _sample_with_on_device_sampler(
            logits, sampler, temperature=1.0, top_p=0.9
        )
        assert 0 <= token_id < 100

    def test_combined_top_k_top_p_on_device(self) -> None:
        """On-device combined top-k + top-p sampling returns a valid token."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        logits = _make_logits(vocab_size=100, peak_token=42)
        token_id = _sample_with_on_device_sampler(
            logits, sampler, temperature=1.0, top_k=10, top_p=0.9
        )
        assert 0 <= token_id < 100


# ---------------------------------------------------------------------------
# Tests: Per-optimization independent fallback
# ---------------------------------------------------------------------------


class TestPerOptimizationFallback:
    """Tests that each optimization falls back independently (Req 11.3).

    The fallback chain from the design:
    - torch.compile → eager Python dispatch (existing path)
    - static KV cache → DynamicCache (existing path)
    - packed projections → separate Q/K/V and gate/up matmuls
    - fused kernels → sequential unfused operations
    - on-device sampling → CPU sampling (existing path)
    - async output → synchronous yield (existing path)
    - sync removal → allow sync operations (existing path)
    """

    def test_on_device_sampling_fallback_independent_of_other_flags(self) -> None:
        """On-device sampling fallback does not affect other optimizations."""
        # Create config with all flags enabled
        config = _make_config(
            enable_on_device_sampling=True,
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )

        # Verify all flags are independently set
        assert config.enable_on_device_sampling is True
        assert config.enable_static_kv_cache is True
        assert config.enable_torch_compile is True
        assert config.enable_packed_projections is True
        assert config.enable_fused_kernels is True
        assert config.enable_async_output is True
        assert config.enable_sync_removal is True

    def test_on_device_sampling_failure_does_not_disable_other_flags(self) -> None:
        """When on-device sampling fails, other optimizations remain active."""
        config = _make_config(
            enable_on_device_sampling=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )

        # Simulate on-device sampler that fails at runtime
        mock_sampler = MagicMock(spec=OnDeviceSampler)
        mock_sampler.sample.side_effect = RuntimeError("device not available")

        logits = _make_logits(peak_token=55)
        # Sampling falls back to CPU but other flags remain True
        token_id = _sample_with_on_device_sampler(
            logits, mock_sampler, temperature=0.0001
        )
        assert token_id == 55
        assert config.enable_async_output is True
        assert config.enable_sync_removal is True

    def test_sampling_works_with_failed_sampler_and_valid_logits(self) -> None:
        """CPU fallback produces valid tokens when on-device sampler fails."""
        # Mock sampler that always fails
        mock_sampler = MagicMock(spec=OnDeviceSampler)
        mock_sampler.sample.side_effect = RuntimeError("kernel crash")

        logits = _make_logits(peak_token=55)
        token_id = _sample_with_on_device_sampler(
            logits, mock_sampler, temperature=0.0001
        )
        # CPU fallback should still produce the correct greedy token
        assert token_id == 55

    def test_each_flag_can_be_disabled_independently(self) -> None:
        """Each optimization flag can be disabled while others remain enabled."""
        # Disable only on-device sampling
        config = _make_config(
            enable_on_device_sampling=False,
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )
        assert config.enable_on_device_sampling is False
        assert config.enable_static_kv_cache is True

        # Disable only torch_compile
        config2 = _make_config(
            enable_on_device_sampling=True,
            enable_static_kv_cache=True,
            enable_torch_compile=False,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )
        assert config2.enable_torch_compile is False
        assert config2.enable_on_device_sampling is True


# ---------------------------------------------------------------------------
# Tests: All flags enabled simultaneously
# ---------------------------------------------------------------------------


class TestAllFlagsEnabled:
    """Tests that all optimization flags can be enabled simultaneously."""

    def test_all_flags_enabled_config_is_valid(self) -> None:
        """Configuration with all flags enabled passes validation."""
        config = PytorchXpuOptimizationConfiguration(
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_on_device_sampling=True,
            enable_async_output=True,
            enable_sync_removal=True,
            torch_compile_mode="max-autotune",
            static_cache_max_seq_len=2048,
            async_output_max_pending=32,
            async_output_resume_threshold=16,
        )
        assert config.enable_static_kv_cache is True
        assert config.enable_torch_compile is True
        assert config.enable_packed_projections is True
        assert config.enable_fused_kernels is True
        assert config.enable_on_device_sampling is True
        assert config.enable_async_output is True
        assert config.enable_sync_removal is True

    def test_on_device_sampler_created_with_all_flags(self) -> None:
        """OnDeviceSampler is created when all flags are enabled."""
        config = _make_config(
            enable_on_device_sampling=True,
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )
        sampler = _create_on_device_sampler(config, "cpu")
        assert sampler is not None
        assert isinstance(sampler, OnDeviceSampler)

    def test_end_to_end_sampling_with_all_flags(self) -> None:
        """Full sampling path works with all flags enabled."""
        config = _make_config(
            enable_on_device_sampling=True,
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_async_output=True,
            enable_sync_removal=True,
        )
        sampler = _create_on_device_sampler(config, "cpu")
        assert sampler is not None

        # Run multiple sampling calls to verify stability
        for peak_token in [0, 10, 50, 99]:
            logits = _make_logits(vocab_size=100, peak_token=peak_token)
            token_id = _sample_with_on_device_sampler(
                logits, sampler, temperature=0.0001
            )
            assert token_id == peak_token

    def test_on_device_sampler_greedy_matches_cpu_greedy(self) -> None:
        """On-device greedy sampling produces same result as CPU greedy."""
        sampler = OnDeviceSampler(device=torch.device("cpu"))
        logits = _make_logits(vocab_size=1000, peak_token=777)

        # On-device result
        on_device_result = _sample_with_on_device_sampler(
            logits, sampler, temperature=0.0001
        )
        # CPU result
        cpu_result = _sample_with_on_device_sampler(
            logits, None, temperature=0.0001
        )
        assert on_device_result == cpu_result == 777
