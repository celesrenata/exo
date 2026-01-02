"""Test IPEX error handling and fallback mechanisms."""

from unittest.mock import Mock, patch

import pytest

from exo.worker.engines.ipex import (
    IPEXDistributedError,
    IPEXDriverError,
    IPEXEngineError,
    IPEXInferenceError,
    IPEXInitializationError,
    IPEXMemoryError,
    IPEXModelLoadError,
)
from exo.worker.engines.ipex.utils_ipex import (
    clear_intel_gpu_memory,
    get_intel_gpu_memory_usage,
    get_ipex_error_context,
    handle_intel_gpu_memory_error,
    handle_ipex_fallback,
    optimize_intel_gpu_memory_usage,
)


class TestIPEXErrorClasses:
    """Test IPEX error class hierarchy and functionality."""

    def test_ipex_engine_error_base(self):
        """Test base IPEX engine error."""
        error = IPEXEngineError("Base error", device_id=0, error_code="BASE_ERROR")

        assert str(error) == "Base error"
        assert error.device_id == 0
        assert error.error_code == "BASE_ERROR"

    def test_ipex_driver_error(self):
        """Test IPEX driver error."""
        error = IPEXDriverError(
            "Driver error", driver_version="1.0.0", required_version="2.0.0"
        )

        assert str(error) == "Driver error"
        assert error.driver_version == "1.0.0"
        assert error.required_version == "2.0.0"
        assert error.error_code == "DRIVER_ERROR"

    def test_ipex_driver_error_defaults(self):
        """Test IPEX driver error with default values."""
        error = IPEXDriverError()

        assert "Intel GPU driver not available or incompatible" in str(error)
        assert error.driver_version is None
        assert error.required_version is None

    def test_ipex_memory_error(self):
        """Test IPEX memory error."""
        error = IPEXMemoryError(
            "Memory error", device_id=1, requested_memory=1024, available_memory=512
        )

        assert str(error) == "Memory error"
        assert error.device_id == 1
        assert error.requested_memory == 1024
        assert error.available_memory == 512
        assert error.error_code == "MEMORY_ERROR"

    def test_ipex_memory_error_defaults(self):
        """Test IPEX memory error with default values."""
        error = IPEXMemoryError()

        assert "Intel GPU memory allocation failed" in str(error)
        assert error.device_id is None
        assert error.requested_memory is None
        assert error.available_memory is None

    def test_ipex_initialization_error(self):
        """Test IPEX initialization error."""
        error = IPEXInitializationError("Init error", component="model_loader")

        assert str(error) == "Init error"
        assert error.component == "model_loader"
        assert error.error_code == "INIT_ERROR"

    def test_ipex_model_load_error(self):
        """Test IPEX model load error."""
        error = IPEXModelLoadError(
            "Model load error", model_path="/path/to/model", device_id=0
        )

        assert str(error) == "Model load error"
        assert error.model_path == "/path/to/model"
        assert error.device_id == 0
        assert error.error_code == "MODEL_LOAD_ERROR"

    def test_ipex_inference_error(self):
        """Test IPEX inference error."""
        error = IPEXInferenceError(
            "Inference error", device_id=0, step="token_generation"
        )

        assert str(error) == "Inference error"
        assert error.device_id == 0
        assert error.step == "token_generation"
        assert error.error_code == "INFERENCE_ERROR"

    def test_ipex_distributed_error(self):
        """Test IPEX distributed error."""
        error = IPEXDistributedError("Distributed error", rank=1, world_size=4)

        assert str(error) == "Distributed error"
        assert error.rank == 1
        assert error.world_size == 4
        assert error.error_code == "DISTRIBUTED_ERROR"


class TestIPEXFallbackHandling:
    """Test IPEX fallback handling functionality."""

    def test_handle_ipex_fallback_driver_error(self):
        """Test fallback handling for driver errors."""
        error = IPEXDriverError("Driver not found")

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache = Mock()

            # Should not raise exception
            handle_ipex_fallback(error, "torch")

            # Verify cache was cleared
            mock_xpu.empty_cache.assert_called_once()

    def test_handle_ipex_fallback_memory_error(self):
        """Test fallback handling for memory errors."""
        error = IPEXMemoryError("Out of memory", device_id=0)

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache = Mock()

            handle_ipex_fallback(error, "cpu")

            mock_xpu.empty_cache.assert_called_once()

    def test_handle_ipex_fallback_model_load_error(self):
        """Test fallback handling for model load errors."""
        error = IPEXModelLoadError("Model incompatible")

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache = Mock()

            handle_ipex_fallback(error, "torch")

            mock_xpu.empty_cache.assert_called_once()

    def test_handle_ipex_fallback_generic_error(self):
        """Test fallback handling for generic errors."""
        error = Exception("Generic error")

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache = Mock()

            handle_ipex_fallback(error, "torch")

            mock_xpu.empty_cache.assert_called_once()

    def test_handle_ipex_fallback_no_xpu(self):
        """Test fallback handling when XPU is not available."""
        error = IPEXDriverError("Driver error")

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = False

            # Should not raise exception even without XPU
            handle_ipex_fallback(error, "cpu")

    def test_handle_ipex_fallback_cleanup_error(self):
        """Test fallback handling when cleanup fails."""
        error = IPEXMemoryError("Memory error")

        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache.side_effect = Exception("Cleanup failed")

            # Should not raise exception even if cleanup fails
            handle_ipex_fallback(error, "torch")


class TestIPEXErrorContext:
    """Test IPEX error context functionality."""

    def test_get_ipex_error_context_base_error(self):
        """Test error context for base IPEX error."""
        error = IPEXEngineError("Base error", device_id=0, error_code="BASE_ERROR")

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {
                "intel_gpu_available": True,
                "intel_gpu_count": 1,
                "ipex_version": "1.13.0",
            }

            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXEngineError"
            assert context["error_message"] == "Base error"
            assert context["error_code"] == "BASE_ERROR"
            assert context["device_id"] == 0
            assert context["intel_gpu_available"] is True
            assert context["intel_gpu_count"] == 1
            assert context["ipex_version"] == "1.13.0"

    def test_get_ipex_error_context_driver_error(self):
        """Test error context for driver error."""
        error = IPEXDriverError(
            "Driver error", driver_version="1.0.0", required_version="2.0.0"
        )

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {"intel_gpu_available": False}

            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXDriverError"
            assert context["driver_version"] == "1.0.0"
            assert context["required_version"] == "2.0.0"

    def test_get_ipex_error_context_memory_error(self):
        """Test error context for memory error."""
        error = IPEXMemoryError(
            "Memory error", device_id=1, requested_memory=1024, available_memory=512
        )

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {"intel_gpu_available": True}

            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXMemoryError"
            assert context["device_id"] == 1
            assert context["requested_memory"] == 1024
            assert context["available_memory"] == 512

    def test_get_ipex_error_context_model_load_error(self):
        """Test error context for model load error."""
        error = IPEXModelLoadError(
            "Model load error", model_path="/path/to/model", device_id=0
        )

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {"intel_gpu_available": True}

            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXModelLoadError"
            assert context["model_path"] == "/path/to/model"
            assert context["device_id"] == 0

    def test_get_ipex_error_context_distributed_error(self):
        """Test error context for distributed error."""
        error = IPEXDistributedError("Distributed error", rank=1, world_size=4)

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {"intel_gpu_available": True}

            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXDistributedError"
            assert context["rank"] == 1
            assert context["world_size"] == 4

    def test_get_ipex_error_context_non_ipex_error(self):
        """Test error context for non-IPEX error."""
        error = ValueError("Generic error")

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info"
        ) as mock_info:
            mock_info.return_value = {"intel_gpu_available": False}

            context = get_ipex_error_context(error)

            assert context["error_type"] == "ValueError"
            assert context["error_message"] == "Generic error"
            assert "error_code" not in context
            assert "device_id" not in context

    def test_get_ipex_error_context_info_failure(self):
        """Test error context when getting Intel GPU info fails."""
        error = IPEXEngineError("Base error")

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info",
            side_effect=Exception("Info failed"),
        ):
            context = get_ipex_error_context(error)

            assert context["error_type"] == "IPEXEngineError"
            assert context["intel_gpu_available"] is False


class TestIPEXMemoryManagement:
    """Test IPEX memory management and error handling."""

    def test_get_intel_gpu_memory_usage_success(self):
        """Test successful Intel GPU memory usage retrieval."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
            mock_xpu.memory_reserved.return_value = 1024 * 1024 * 200  # 200MB

            # Mock device properties
            mock_props = Mock()
            mock_props.total_memory = 1024 * 1024 * 1024 * 8  # 8GB
            mock_xpu.get_device_properties.return_value = mock_props

            memory_info = get_intel_gpu_memory_usage(0)

            assert memory_info["allocated"] == 1024 * 1024 * 100
            assert memory_info["reserved"] == 1024 * 1024 * 200
            assert memory_info["total"] == 1024 * 1024 * 1024 * 8
            assert memory_info["free"] > 0

    def test_get_intel_gpu_memory_usage_no_xpu(self):
        """Test Intel GPU memory usage when XPU is not available."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = False

            memory_info = get_intel_gpu_memory_usage(0)

            assert memory_info["allocated"] == 0
            assert memory_info["reserved"] == 0
            assert memory_info["free"] == 0
            assert memory_info["total"] == 0

    def test_get_intel_gpu_memory_usage_invalid_device(self):
        """Test Intel GPU memory usage with invalid device ID."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1

            # Request device 1 when only device 0 exists
            memory_info = get_intel_gpu_memory_usage(1)

            assert memory_info["allocated"] == 0
            assert memory_info["reserved"] == 0
            assert memory_info["free"] == 0
            assert memory_info["total"] == 0

    def test_clear_intel_gpu_memory_success(self):
        """Test successful Intel GPU memory clearing."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache = Mock()

            clear_intel_gpu_memory(0)

            mock_xpu.empty_cache.assert_called_once()

    def test_clear_intel_gpu_memory_no_xpu(self):
        """Test Intel GPU memory clearing when XPU is not available."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = False

            # Should not raise exception
            clear_intel_gpu_memory(0)

    def test_clear_intel_gpu_memory_error(self):
        """Test Intel GPU memory clearing with error."""
        with patch("torch.xpu") as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.empty_cache.side_effect = Exception("Clear failed")

            # Should not raise exception
            clear_intel_gpu_memory(0)

    def test_optimize_intel_gpu_memory_usage_success(self):
        """Test successful Intel GPU memory optimization."""
        with (
            patch("gc.collect") as mock_gc,
            patch("torch.xpu") as mock_xpu,
            patch(
                "exo.worker.engines.ipex.utils_ipex.clear_intel_gpu_memory"
            ) as mock_clear,
        ):
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 2

            optimize_intel_gpu_memory_usage()

            mock_gc.assert_called_once()
            assert mock_clear.call_count == 2  # Called for each device

    def test_optimize_intel_gpu_memory_usage_error(self):
        """Test Intel GPU memory optimization with error."""
        with patch("gc.collect", side_effect=Exception("GC failed")):
            # Should not raise exception
            optimize_intel_gpu_memory_usage()

    def test_handle_intel_gpu_memory_error_recovery_success(self):
        """Test successful Intel GPU memory error recovery."""
        error = Exception("Memory error")

        with (
            patch(
                "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage"
            ) as mock_memory,
            patch(
                "exo.worker.engines.ipex.utils_ipex.optimize_intel_gpu_memory_usage"
            ) as mock_optimize,
        ):
            # Mock memory info before and after optimization
            mock_memory.side_effect = [
                {"free": 1024 * 1024 * 100},  # Before: 100MB free
                {"free": 1024 * 1024 * 200},  # After: 200MB free
            ]

            result = handle_intel_gpu_memory_error(error, 0)

            assert result is True
            mock_optimize.assert_called_once()

    def test_handle_intel_gpu_memory_error_recovery_failure(self):
        """Test Intel GPU memory error recovery failure."""
        error = Exception("Memory error")

        with (
            patch(
                "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage"
            ) as mock_memory,
            patch(
                "exo.worker.engines.ipex.utils_ipex.optimize_intel_gpu_memory_usage"
            ) as mock_optimize,
        ):
            # Mock no memory recovery
            mock_memory.side_effect = [
                {"free": 1024 * 1024 * 100},  # Before: 100MB free
                {"free": 1024 * 1024 * 100},  # After: still 100MB free
            ]

            with pytest.raises(IPEXMemoryError) as exc_info:
                handle_intel_gpu_memory_error(error, 0)

            assert "Intel GPU memory recovery failed" in str(exc_info.value)
            mock_optimize.assert_called_once()

    def test_handle_intel_gpu_memory_error_exception_during_recovery(self):
        """Test Intel GPU memory error when recovery itself fails."""
        error = Exception("Memory error")

        with patch(
            "exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage",
            side_effect=Exception("Memory info failed"),
        ):
            with pytest.raises(IPEXMemoryError) as exc_info:
                handle_intel_gpu_memory_error(error, 0)

            assert "Intel GPU memory error handling failed" in str(exc_info.value)
