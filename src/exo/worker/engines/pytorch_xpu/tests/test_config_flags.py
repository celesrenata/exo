"""
Unit tests for PytorchXpuOptimizationConfiguration kernel launch optimization flags.

Tests verify:
- All new flags default to False
- Configuration immutability (frozen=True)
- Validators for numeric parameters
"""

from __future__ import annotations

import pytest

from exo.worker.engines.pytorch_xpu.pipeline_config import (
    PytorchXpuOptimizationConfiguration,
)


class TestOptimizationFlagsDefaults:
    """Test that all kernel launch optimization flags default to False."""

    def test_enable_static_kv_cache_defaults_to_false(self) -> None:
        """Verify enable_static_kv_cache defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_static_kv_cache is False

    def test_enable_torch_compile_defaults_to_false(self) -> None:
        """Verify enable_torch_compile defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_torch_compile is False

    def test_enable_packed_projections_defaults_to_false(self) -> None:
        """Verify enable_packed_projections defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_packed_projections is False

    def test_enable_fused_kernels_defaults_to_false(self) -> None:
        """Verify enable_fused_kernels defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_fused_kernels is False

    def test_enable_on_device_sampling_defaults_to_false(self) -> None:
        """Verify enable_on_device_sampling defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_on_device_sampling is False

    def test_enable_async_output_defaults_to_false(self) -> None:
        """Verify enable_async_output defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_async_output is False

    def test_enable_sync_removal_defaults_to_false(self) -> None:
        """Verify enable_sync_removal defaults to False."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.enable_sync_removal is False


class TestNumericFlagDefaults:
    """Test default values for numeric configuration parameters."""

    def test_torch_compile_mode_defaults_to_max_autotune(self) -> None:
        """Verify torch_compile_mode defaults to 'max-autotune'."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.torch_compile_mode == "max-autotune"

    def test_static_cache_max_seq_len_defaults_to_2048(self) -> None:
        """Verify static_cache_max_seq_len defaults to 2048."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.static_cache_max_seq_len == 2048

    def test_async_output_max_pending_defaults_to_32(self) -> None:
        """Verify async_output_max_pending defaults to 32."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.async_output_max_pending == 32

    def test_async_output_resume_threshold_defaults_to_16(self) -> None:
        """Verify async_output_resume_threshold defaults to 16."""
        config = PytorchXpuOptimizationConfiguration()
        assert config.async_output_resume_threshold == 16


class TestConfigurationImmutability:
    """Test that configuration is immutable (frozen=True)."""

    def test_cannot_modify_enable_static_kv_cache(self) -> None:
        """Verify enable_static_kv_cache cannot be modified after creation."""
        config = PytorchXpuOptimizationConfiguration()
        with pytest.raises(Exception):  # pydantic.errors.FrozenError
            config.enable_static_kv_cache = True

    def test_cannot_modify_enable_torch_compile(self) -> None:
        """Verify enable_torch_compile cannot be modified after creation."""
        config = PytorchXpuOptimizationConfiguration()
        with pytest.raises(Exception):  # pydantic.errors.FrozenError
            config.enable_torch_compile = True

    def test_cannot_modify_torch_compile_mode(self) -> None:
        """Verify torch_compile_mode cannot be modified after creation."""
        config = PytorchXpuOptimizationConfiguration()
        with pytest.raises(Exception):  # pydantic.errors.FrozenError
            config.torch_compile_mode = "default"

    def test_cannot_modify_static_cache_max_seq_len(self) -> None:
        """Verify static_cache_max_seq_len cannot be modified after creation."""
        config = PytorchXpuOptimizationConfiguration()
        with pytest.raises(Exception):  # pydantic.errors.FrozenError
            config.static_cache_max_seq_len = 1024


class TestNumericValidators:
    """Test validators for numeric configuration parameters."""

    def test_torch_compile_mode_accepts_valid_values(self) -> None:
        """Verify torch_compile_mode accepts valid mode strings."""
        for mode in ("default", "reduce-overhead", "max-autotune"):
            config = PytorchXpuOptimizationConfiguration(
                torch_compile_mode=mode
            )
            assert config.torch_compile_mode == mode

    def test_torch_compile_mode_rejects_invalid_values(self) -> None:
        """Verify torch_compile_mode rejects invalid mode strings."""
        with pytest.raises(Exception) as exc_info:
            PytorchXpuOptimizationConfiguration(
                torch_compile_mode="invalid-mode"  # pyright: ignore[reportArgumentType]
            )
        assert "invalid-mode" in str(exc_info.value)

    def test_static_cache_max_seq_len_accepts_positive_values(self) -> None:
        """Verify static_cache_max_seq_len accepts positive values."""
        for value in (1, 512, 1024, 2048, 4096):
            config = PytorchXpuOptimizationConfiguration(
                static_cache_max_seq_len=value
            )
            assert config.static_cache_max_seq_len == value

    def test_static_cache_max_seq_len_rejects_zero(self) -> None:
        """Verify static_cache_max_seq_len rejects zero."""
        with pytest.raises(ValueError) as exc_info:
            PytorchXpuOptimizationConfiguration(
                static_cache_max_seq_len=0
            )
        assert "static_cache_max_seq_len must be positive" in str(exc_info.value)

    def test_static_cache_max_seq_len_rejects_negative(self) -> None:
        """Verify static_cache_max_seq_len rejects negative values."""
        with pytest.raises(ValueError) as exc_info:
            PytorchXpuOptimizationConfiguration(
                static_cache_max_seq_len=-1
            )
        assert "static_cache_max_seq_len must be positive" in str(exc_info.value)

    def test_async_output_max_pending_accepts_positive_values(self) -> None:
        """Verify async_output_max_pending accepts positive values."""
        for value in (1, 16, 32, 64, 128):
            config = PytorchXpuOptimizationConfiguration(
                async_output_max_pending=value
            )
            assert config.async_output_max_pending == value

    def test_async_output_max_pending_rejects_zero(self) -> None:
        """Verify async_output_max_pending rejects zero."""
        with pytest.raises(ValueError) as exc_info:
            PytorchXpuOptimizationConfiguration(
                async_output_max_pending=0
            )
        assert "async_output_max_pending must be positive" in str(exc_info.value)

    def test_async_output_resume_threshold_accepts_positive_values(self) -> None:
        """Verify async_output_resume_threshold accepts positive values."""
        for value in (1, 8, 16, 32, 64):
            config = PytorchXpuOptimizationConfiguration(
                async_output_resume_threshold=value
            )
            assert config.async_output_resume_threshold == value

    def test_async_output_resume_threshold_rejects_zero(self) -> None:
        """Verify async_output_resume_threshold rejects zero."""
        with pytest.raises(ValueError) as exc_info:
            PytorchXpuOptimizationConfiguration(
                async_output_resume_threshold=0
            )
        assert "async_output_resume_threshold must be positive" in str(exc_info.value)


class TestConfigurationWithCustomValues:
    """Test configuration with custom flag values."""

    def test_all_flags_enabled(self) -> None:
        """Verify configuration works with all flags set to True."""
        config = PytorchXpuOptimizationConfiguration(
            enable_static_kv_cache=True,
            enable_torch_compile=True,
            enable_packed_projections=True,
            enable_fused_kernels=True,
            enable_on_device_sampling=True,
            enable_async_output=True,
            enable_sync_removal=True,
            torch_compile_mode="default",
            static_cache_max_seq_len=1024,
            async_output_max_pending=64,
            async_output_resume_threshold=32,
        )
        assert config.enable_static_kv_cache is True
        assert config.enable_torch_compile is True
        assert config.enable_packed_projections is True
        assert config.enable_fused_kernels is True
        assert config.enable_on_device_sampling is True
        assert config.enable_async_output is True
        assert config.enable_sync_removal is True
        assert config.torch_compile_mode == "default"
        assert config.static_cache_max_seq_len == 1024
        assert config.async_output_max_pending == 64
        assert config.async_output_resume_threshold == 32

    def test_mixed_flags(self) -> None:
        """Verify configuration works with some flags enabled and others disabled."""
        config = PytorchXpuOptimizationConfiguration(
            enable_static_kv_cache=True,
            enable_torch_compile=False,
            enable_packed_projections=True,
            enable_fused_kernels=False,
            enable_on_device_sampling=True,
            enable_async_output=False,
            enable_sync_removal=True,
        )
        assert config.enable_static_kv_cache is True
        assert config.enable_torch_compile is False
        assert config.enable_packed_projections is True
        assert config.enable_fused_kernels is False
        assert config.enable_on_device_sampling is True
        assert config.enable_async_output is False
        assert config.enable_sync_removal is True
