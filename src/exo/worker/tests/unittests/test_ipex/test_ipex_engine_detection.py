"""Test Intel GPU detection and engine selection logic."""

from unittest.mock import Mock, patch

import pytest

from exo.worker.engines.engine_utils import (
    detect_available_engines,
    detect_intel_gpu,
    select_best_engine,
)
from exo.worker.engines.ipex import IPEXDriverError, IPEXInitializationError
from exo.worker.engines.ipex.utils_ipex import (
    check_ipex_availability,
    validate_intel_gpu_environment,
)


class TestIntelGPUDetection:
    """Test Intel GPU detection functionality."""

    def test_detect_intel_gpu_success(self):
        """Test successful Intel GPU detection."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            # Mock successful Intel GPU detection
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Mock tensor operations
            mock_device = Mock()
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())

            with (
                patch("torch.device", return_value=mock_device),
                patch("torch.tensor", return_value=mock_tensor),
            ):
                result = detect_intel_gpu()
                assert result is True

                # Verify calls
                mock_xpu.is_available.assert_called_once()
                mock_xpu.device_count.assert_called_once()

    def test_detect_intel_gpu_no_ipex(self):
        """Test Intel GPU detection when IPEX is not installed."""
        with patch(
            "intel_extension_for_pytorch",
            side_effect=ImportError("No module named 'intel_extension_for_pytorch'"),
        ):
            result = detect_intel_gpu()
            assert result is False

    def test_detect_intel_gpu_no_xpu(self):
        """Test Intel GPU detection when XPU is not available."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = False

            result = detect_intel_gpu()
            assert result is False

    def test_detect_intel_gpu_no_devices(self):
        """Test Intel GPU detection when no devices are found."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 0

            result = detect_intel_gpu()
            assert result is False

    def test_detect_intel_gpu_tensor_operation_failure(self):
        """Test Intel GPU detection when tensor operations fail."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Mock tensor operation failure
            with (
                patch("torch.device"),
                patch(
                    "torch.tensor",
                    side_effect=RuntimeError("Intel GPU operation failed"),
                ),
            ):
                result = detect_intel_gpu()
                assert result is False


class TestEngineSelection:
    """Test engine selection logic with IPEX."""

    def test_detect_available_engines_with_ipex(self):
        """Test engine detection includes IPEX when available."""
        with (
            patch(
                "exo.worker.engines.engine_utils.detect_intel_gpu", return_value=True
            ),
            patch("mlx.core") as mock_mlx,
            patch("torch") as mock_torch,
        ):
            # Mock MLX not available
            mock_mlx.metal.is_available.return_value = False

            # Mock torch available
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())
            mock_torch.tensor.return_value = mock_tensor

            engines = detect_available_engines()

            assert "ipex" in engines
            assert "torch" in engines
            assert "cpu" in engines

    def test_detect_available_engines_without_ipex(self):
        """Test engine detection when IPEX is not available."""
        with (
            patch(
                "exo.worker.engines.engine_utils.detect_intel_gpu", return_value=False
            ),
            patch("mlx.core") as mock_mlx,
            patch("torch") as mock_torch,
        ):
            # Mock MLX not available
            mock_mlx.metal.is_available.return_value = False

            # Mock torch available
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())
            mock_torch.tensor.return_value = mock_tensor

            engines = detect_available_engines()

            assert "ipex" not in engines
            assert "torch" in engines
            assert "cpu" in engines

    def test_select_best_engine_prefers_ipex_over_torch(self):
        """Test that IPEX is preferred over torch when both are available."""
        with patch(
            "exo.worker.engines.engine_utils.detect_available_engines"
        ) as mock_detect:
            mock_detect.return_value = ["ipex", "torch", "cpu"]

            engine = select_best_engine()
            assert engine == "ipex"

    def test_select_best_engine_mlx_preferred_over_ipex(self):
        """Test that MLX is preferred over IPEX when both are available."""
        with patch(
            "exo.worker.engines.engine_utils.detect_available_engines"
        ) as mock_detect:
            mock_detect.return_value = ["mlx", "ipex", "torch", "cpu"]

            engine = select_best_engine()
            assert engine == "mlx"

    def test_select_best_engine_forced_ipex(self):
        """Test forced IPEX engine selection via environment variable."""
        with (
            patch(
                "exo.worker.engines.engine_utils.detect_available_engines"
            ) as mock_detect,
            patch("os.getenv", return_value="ipex"),
        ):
            mock_detect.return_value = ["mlx", "ipex", "torch", "cpu"]

            engine = select_best_engine()
            assert engine == "ipex"

    def test_select_best_engine_forced_unavailable(self):
        """Test forced engine selection when forced engine is not available."""
        with (
            patch(
                "exo.worker.engines.engine_utils.detect_available_engines"
            ) as mock_detect,
            patch("os.getenv", return_value="ipex"),
        ):
            mock_detect.return_value = ["mlx", "torch", "cpu"]  # IPEX not available

            engine = select_best_engine()
            assert engine == "mlx"  # Falls back to best available


class TestIPEXAvailabilityCheck:
    """Test IPEX availability checking functionality."""

    def test_check_ipex_availability_success(self):
        """Test successful IPEX availability check."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 2

            # Mock tensor operations
            mock_device = Mock()
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())

            with (
                patch("torch.device", return_value=mock_device),
                patch("torch.tensor", return_value=mock_tensor),
            ):
                result = check_ipex_availability()
                assert result is True

    def test_check_ipex_availability_no_ipex(self):
        """Test IPEX availability check when IPEX is not installed."""
        with patch(
            "intel_extension_for_pytorch", side_effect=ImportError("IPEX not found")
        ):
            result = check_ipex_availability()
            assert result is False

    def test_check_ipex_availability_no_xpu(self):
        """Test IPEX availability check when XPU is not available."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = False

            result = check_ipex_availability()
            assert result is False

    def test_check_ipex_availability_tensor_failure(self):
        """Test IPEX availability check when tensor operations fail."""
        with (
            patch("intel_extension_for_pytorch"),
            patch("torch.xpu") as mock_xpu,
        ):
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Mock tensor operation failure
            with (
                patch("torch.device"),
                patch(
                    "torch.tensor", side_effect=RuntimeError("Tensor operation failed")
                ),
            ):
                result = check_ipex_availability()
                assert result is False


class TestIPEXEnvironmentValidation:
    """Test IPEX environment validation functionality."""

    def test_validate_intel_gpu_environment_success(self):
        """Test successful Intel GPU environment validation."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            # Mock IPEX version
            mock_ipex.__version__ = "1.13.0"

            # Mock Intel GPU availability
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Mock successful device operations
            mock_device = Mock()
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())

            with (
                patch("torch.device", return_value=mock_device),
                patch("torch.tensor", return_value=mock_tensor),
            ):
                # Should not raise any exception
                validate_intel_gpu_environment()

    def test_validate_intel_gpu_environment_no_ipex(self):
        """Test environment validation when IPEX is not installed."""
        with patch(
            "intel_extension_for_pytorch", side_effect=ImportError("IPEX not found")
        ):
            with pytest.raises(IPEXDriverError) as exc_info:
                validate_intel_gpu_environment()

            assert "Intel Extension for PyTorch not installed" in str(exc_info.value)

    def test_validate_intel_gpu_environment_corrupted_ipex(self):
        """Test environment validation when IPEX installation is corrupted."""
        with patch("intel_extension_for_pytorch") as mock_ipex:
            # Mock corrupted installation (no version attribute)
            del mock_ipex.__version__

            with pytest.raises(IPEXDriverError) as exc_info:
                validate_intel_gpu_environment()

            assert "Intel Extension for PyTorch installation is corrupted" in str(
                exc_info.value
            )

    def test_validate_intel_gpu_environment_no_xpu(self):
        """Test environment validation when XPU is not available."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            mock_ipex.__version__ = "1.13.0"
            mock_xpu.is_available.return_value = False

            with pytest.raises(IPEXDriverError) as exc_info:
                validate_intel_gpu_environment()

            assert "Intel GPU (XPU) not available" in str(exc_info.value)

    def test_validate_intel_gpu_environment_no_devices(self):
        """Test environment validation when no Intel GPU devices are found."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            mock_ipex.__version__ = "1.13.0"
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 0

            with pytest.raises(IPEXDriverError) as exc_info:
                validate_intel_gpu_environment()

            assert "No Intel GPU devices found" in str(exc_info.value)

    def test_validate_intel_gpu_environment_device_validation_failure(self):
        """Test environment validation when device validation fails."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            mock_ipex.__version__ = "1.13.0"
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Mock device validation failure
            with (
                patch("torch.device"),
                patch(
                    "torch.tensor", side_effect=RuntimeError("Device validation failed")
                ),
            ):
                with pytest.raises(IPEXDriverError) as exc_info:
                    validate_intel_gpu_environment()

                assert "Intel GPU device 0 failed validation" in str(exc_info.value)

    def test_validate_intel_gpu_environment_multiple_devices(self):
        """Test environment validation with multiple Intel GPU devices."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            mock_ipex.__version__ = "1.13.0"
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 2

            # Mock successful device operations for both devices
            mock_device = Mock()
            mock_tensor = Mock()
            mock_tensor.__add__ = Mock(return_value=Mock())

            with (
                patch("torch.device", return_value=mock_device),
                patch("torch.tensor", return_value=mock_tensor),
            ):
                # Should not raise any exception
                validate_intel_gpu_environment()

    def test_validate_intel_gpu_environment_unexpected_error(self):
        """Test environment validation with unexpected error."""
        with (
            patch("intel_extension_for_pytorch") as mock_ipex,
            patch("torch.xpu") as mock_xpu,
        ):
            mock_ipex.__version__ = "1.13.0"
            mock_xpu.is_available.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(IPEXInitializationError) as exc_info:
                validate_intel_gpu_environment()

            assert "Intel GPU environment validation failed" in str(exc_info.value)
