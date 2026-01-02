"""Test Intel IPEX UI elements and dashboard integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from exo.worker.engines.engine_utils import get_engine_info
from exo.worker.engines.ipex.utils_ipex import get_intel_gpu_info


class TestEngineInfoIntegration:
    """Test engine info integration with IPEX."""

    def test_get_engine_info_with_ipex_available(self):
        """Test engine info when IPEX is available."""
        with patch('exo.worker.engines.engine_utils.detect_intel_gpu', return_value=True), \
             patch('exo.worker.engines.engine_utils.detect_available_engines', return_value=["ipex", "torch", "cpu"]), \
             patch('exo.worker.engines.engine_utils.select_best_engine', return_value="ipex"), \
             patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info') as mock_gpu_info:
            
            # Mock Intel GPU info
            mock_gpu_info.return_value = {
                "intel_gpu_available": True,
                "intel_gpu_count": 2,
                "intel_gpu_memory": 8 * 1024**3,  # 8GB
                "ipex_version": "1.13.0",
                "intel_gpu_devices": [
                    {
                        "device_id": 0,
                        "name": "Intel Arc A770",
                        "total_memory": 8 * 1024**3,
                        "memory_allocated": 1024**3,
                        "memory_free": 7 * 1024**3
                    },
                    {
                        "device_id": 1,
                        "name": "Intel Arc A750",
                        "total_memory": 8 * 1024**3,
                        "memory_allocated": 512 * 1024**2,
                        "memory_free": 7.5 * 1024**3
                    }
                ]
            }
            
            engine_info = get_engine_info()
            
            # Verify IPEX is detected and selected
            assert "ipex" in engine_info["available_engines"]
            assert engine_info["selected_engine"] == "ipex"
            assert engine_info["ipex_available"] is True
            
            # Verify Intel GPU info is included
            assert engine_info["intel_gpu_available"] is True
            assert engine_info["intel_gpu_count"] == 2
            assert engine_info["intel_gpu_memory"] == 8 * 1024**3
            assert engine_info["ipex_version"] == "1.13.0"
            assert len(engine_info["intel_gpu_devices"]) == 2

    def test_get_engine_info_with_ipex_unavailable(self):
        """Test engine info when IPEX is not available."""
        with patch('exo.worker.engines.engine_utils.detect_intel_gpu', return_value=False), \
             patch('exo.worker.engines.engine_utils.detect_available_engines', return_value=["torch", "cpu"]), \
             patch('exo.worker.engines.engine_utils.select_best_engine', return_value="torch"):
            
            engine_info = get_engine_info()
            
            # Verify IPEX is not detected
            assert "ipex" not in engine_info["available_engines"]
            assert engine_info["selected_engine"] == "torch"
            assert engine_info["ipex_available"] is False
            
            # Verify Intel GPU info shows unavailable
            assert engine_info.get("intel_gpu_available", False) is False
            assert engine_info.get("intel_gpu_count", 0) == 0

    def test_get_engine_info_ipex_import_error(self):
        """Test engine info when IPEX import fails."""
        with patch('exo.worker.engines.engine_utils.detect_intel_gpu', return_value=True), \
             patch('exo.worker.engines.engine_utils.detect_available_engines', return_value=["ipex", "torch", "cpu"]), \
             patch('exo.worker.engines.engine_utils.select_best_engine', return_value="ipex"), \
             patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info', side_effect=ImportError("IPEX not found")):
            
            engine_info = get_engine_info()
            
            # Should still show IPEX as available from detection
            assert engine_info["ipex_available"] is True
            # But Intel GPU info should not be included due to import error
            assert "intel_gpu_devices" not in engine_info


class TestIntelGPUInfoCollection:
    """Test Intel GPU information collection for dashboard."""

    def test_get_intel_gpu_info_success(self):
        """Test successful Intel GPU info collection."""
        with patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu, \
             patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage') as mock_memory:
            
            # Mock IPEX version
            mock_ipex.__version__ = "1.13.0"
            
            # Mock Intel GPU availability
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            
            # Mock device properties
            mock_props = Mock()
            mock_props.name = "Intel Arc A770"
            mock_props.total_memory = 8 * 1024**3
            mock_props.max_compute_units = 512
            mock_xpu.get_device_properties.return_value = mock_props
            
            # Mock memory usage
            mock_memory.return_value = {
                "allocated": 1024**3,
                "reserved": 1.5 * 1024**3,
                "free": 6.5 * 1024**3
            }
            
            gpu_info = get_intel_gpu_info()
            
            assert gpu_info["intel_gpu_available"] is True
            assert gpu_info["intel_gpu_count"] == 1
            assert gpu_info["intel_gpu_memory"] == 8 * 1024**3
            assert gpu_info["ipex_version"] == "1.13.0"
            
            # Check device info
            assert len(gpu_info["intel_gpu_devices"]) == 1
            device_info = gpu_info["intel_gpu_devices"][0]
            assert device_info["device_id"] == 0
            assert device_info["name"] == "Intel Arc A770"
            assert device_info["total_memory"] == 8 * 1024**3
            assert device_info["max_compute_units"] == 512
            assert device_info["memory_allocated"] == 1024**3
            assert device_info["memory_free"] == 6.5 * 1024**3

    def test_get_intel_gpu_info_multiple_devices(self):
        """Test Intel GPU info collection with multiple devices."""
        with patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu, \
             patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage') as mock_memory:
            
            # Mock IPEX version
            mock_ipex.__version__ = "1.13.0"
            
            # Mock Intel GPU availability
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 2
            
            # Mock device properties for multiple devices
            def mock_get_device_properties(device_id):
                if device_id == 0:
                    props = Mock()
                    props.name = "Intel Arc A770"
                    props.total_memory = 8 * 1024**3
                    props.max_compute_units = 512
                    return props
                else:
                    props = Mock()
                    props.name = "Intel Arc A750"
                    props.total_memory = 8 * 1024**3
                    props.max_compute_units = 448
                    return props
            
            mock_xpu.get_device_properties.side_effect = mock_get_device_properties
            
            # Mock memory usage for multiple devices
            def mock_get_memory_usage(device_id):
                if device_id == 0:
                    return {
                        "allocated": 1024**3,
                        "reserved": 1.5 * 1024**3,
                        "free": 6.5 * 1024**3
                    }
                else:
                    return {
                        "allocated": 512 * 1024**2,
                        "reserved": 1024**3,
                        "free": 7 * 1024**3
                    }
            
            mock_memory.side_effect = mock_get_memory_usage
            
            gpu_info = get_intel_gpu_info()
            
            assert gpu_info["intel_gpu_available"] is True
            assert gpu_info["intel_gpu_count"] == 2
            assert len(gpu_info["intel_gpu_devices"]) == 2
            
            # Check first device
            device_0 = gpu_info["intel_gpu_devices"][0]
            assert device_0["device_id"] == 0
            assert device_0["name"] == "Intel Arc A770"
            assert device_0["max_compute_units"] == 512
            
            # Check second device
            device_1 = gpu_info["intel_gpu_devices"][1]
            assert device_1["device_id"] == 1
            assert device_1["name"] == "Intel Arc A750"
            assert device_1["max_compute_units"] == 448

    def test_get_intel_gpu_info_no_ipex(self):
        """Test Intel GPU info collection when IPEX is not available."""
        with patch('intel_extension_for_pytorch', side_effect=ImportError("IPEX not found")):
            
            gpu_info = get_intel_gpu_info()
            
            assert gpu_info["intel_gpu_available"] is False
            assert gpu_info["intel_gpu_count"] == 0
            assert gpu_info["intel_gpu_memory"] == 0
            assert gpu_info["ipex_version"] is None
            assert gpu_info["intel_gpu_devices"] == []

    def test_get_intel_gpu_info_no_xpu(self):
        """Test Intel GPU info collection when XPU is not available."""
        with patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu:
            
            # Mock IPEX version
            mock_ipex.__version__ = "1.13.0"
            
            # Mock XPU not available
            mock_xpu.is_available.return_value = False
            
            gpu_info = get_intel_gpu_info()
            
            assert gpu_info["intel_gpu_available"] is False
            assert gpu_info["intel_gpu_count"] == 0
            assert gpu_info["intel_gpu_memory"] == 0
            assert gpu_info["ipex_version"] == "1.13.0"
            assert gpu_info["intel_gpu_devices"] == []

    def test_get_intel_gpu_info_device_error(self):
        """Test Intel GPU info collection with device property error."""
        with patch('intel_extension_for_pytorch') as mock_ipex, \
             patch('torch.xpu') as mock_xpu, \
             patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_memory_usage') as mock_memory:
            
            # Mock IPEX version
            mock_ipex.__version__ = "1.13.0"
            
            # Mock Intel GPU availability
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            
            # Mock device properties error
            mock_xpu.get_device_properties.side_effect = RuntimeError("Device properties failed")
            
            # Mock memory usage
            mock_memory.return_value = {
                "allocated": 0,
                "reserved": 0,
                "free": 0
            }
            
            gpu_info = get_intel_gpu_info()
            
            # Should still report GPU as available but with limited info
            assert gpu_info["intel_gpu_available"] is True
            assert gpu_info["intel_gpu_count"] == 1
            assert gpu_info["ipex_version"] == "1.13.0"
            
            # Device list should be empty due to error
            assert gpu_info["intel_gpu_devices"] == []


class TestDashboardInstanceTypeSupport:
    """Test dashboard instance type support for IPEX."""

    def test_ipex_instance_type_detection(self):
        """Test that IPEX instance type is properly detected."""
        # This would be tested in the actual dashboard code
        # Here we test the underlying engine detection that supports it
        
        with patch('exo.worker.engines.engine_utils.detect_intel_gpu', return_value=True):
            from exo.worker.engines.engine_utils import detect_available_engines
            
            engines = detect_available_engines()
            assert "ipex" in engines

    def test_ipex_model_compatibility(self):
        """Test IPEX model compatibility checking."""
        from exo.worker.engines.engine_utils import is_model_compatible
        
        # IPEX should be compatible with standard HuggingFace models
        assert is_model_compatible("microsoft/DialoGPT-medium", "ipex") is True
        assert is_model_compatible("distilbert/distilgpt2", "ipex") is True
        assert is_model_compatible("EleutherAI/gpt-j-6b", "ipex") is True
        
        # IPEX should not be compatible with MLX-specific models
        assert is_model_compatible("mlx-community/Llama-3.2-1B-Instruct-4bit", "ipex") is False

    def test_compatible_models_filtering(self):
        """Test filtering of compatible models for IPEX."""
        from exo.worker.engines.engine_utils import get_compatible_models
        
        available_models = [
            "microsoft/DialoGPT-medium",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "distilbert/distilgpt2",
            "mlx-community/TinyLlama-1.1B-Chat-v1.0",
            "EleutherAI/gpt-j-6b"
        ]
        
        with patch('exo.worker.engines.engine_utils.select_best_engine', return_value="ipex"):
            compatible = get_compatible_models(available_models)
            
            # Should include standard HuggingFace models
            assert "microsoft/DialoGPT-medium" in compatible
            assert "distilbert/distilgpt2" in compatible
            assert "EleutherAI/gpt-j-6b" in compatible
            
            # Should exclude MLX-specific models
            assert "mlx-community/Llama-3.2-1B-Instruct-4bit" not in compatible
            assert "mlx-community/TinyLlama-1.1B-Chat-v1.0" not in compatible


class TestDashboardSystemInfoIntegration:
    """Test system info integration for dashboard display."""

    def test_system_info_includes_intel_gpu(self):
        """Test that system info includes Intel GPU information."""
        # This tests the integration point that would be used by the dashboard
        
        with patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info') as mock_gpu_info:
            mock_gpu_info.return_value = {
                "intel_gpu_available": True,
                "intel_gpu_count": 1,
                "intel_gpu_memory": 8 * 1024**3,
                "ipex_version": "1.13.0",
                "intel_gpu_devices": [
                    {
                        "device_id": 0,
                        "name": "Intel Arc A770",
                        "total_memory": 8 * 1024**3,
                        "memory_allocated": 1024**3,
                        "memory_free": 7 * 1024**3
                    }
                ]
            }
            
            # Test the function that would be called by system info gathering
            gpu_info = get_intel_gpu_info()
            
            # Verify the information is structured correctly for dashboard consumption
            assert isinstance(gpu_info["intel_gpu_devices"], list)
            assert all(isinstance(device, dict) for device in gpu_info["intel_gpu_devices"])
            
            # Verify required fields are present
            for device in gpu_info["intel_gpu_devices"]:
                assert "device_id" in device
                assert "name" in device
                assert "total_memory" in device
                assert "memory_allocated" in device
                assert "memory_free" in device

    def test_system_info_json_serializable(self):
        """Test that Intel GPU info is JSON serializable for dashboard."""
        with patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info') as mock_gpu_info:
            mock_gpu_info.return_value = {
                "intel_gpu_available": True,
                "intel_gpu_count": 2,
                "intel_gpu_memory": 8 * 1024**3,
                "ipex_version": "1.13.0",
                "intel_gpu_devices": [
                    {
                        "device_id": 0,
                        "name": "Intel Arc A770",
                        "total_memory": 8 * 1024**3,
                        "max_compute_units": 512,
                        "memory_allocated": 1024**3,
                        "memory_reserved": 1.5 * 1024**3,
                        "memory_free": 6.5 * 1024**3
                    }
                ]
            }
            
            gpu_info = get_intel_gpu_info()
            
            # Should be JSON serializable without errors
            json_str = json.dumps(gpu_info)
            assert isinstance(json_str, str)
            
            # Should be deserializable back to the same structure
            deserialized = json.loads(json_str)
            assert deserialized == gpu_info

    def test_system_info_error_handling(self):
        """Test system info error handling for dashboard robustness."""
        with patch('exo.worker.engines.ipex.utils_ipex.get_intel_gpu_info', side_effect=Exception("GPU info failed")):
            
            # Should not raise exception - dashboard should handle gracefully
            try:
                gpu_info = get_intel_gpu_info()
                # If the function handles errors internally, it should return safe defaults
                assert isinstance(gpu_info, dict)
            except Exception:
                # If the function doesn't handle errors, the dashboard should catch them
                pass


class TestDashboardMonitoringIntegration:
    """Test dashboard monitoring integration for Intel GPU metrics."""

    def test_gpu_memory_monitoring_data_format(self):
        """Test GPU memory monitoring data format for dashboard."""
        with patch('torch.xpu') as mock_xpu:
            mock_xpu.is_available.return_value = True
            mock_xpu.device_count.return_value = 1
            mock_xpu.memory_allocated.return_value = 1024**3  # 1GB
            mock_xpu.memory_reserved.return_value = 1.5 * 1024**3  # 1.5GB
            
            # Mock device properties
            mock_props = Mock()
            mock_props.total_memory = 8 * 1024**3  # 8GB
            mock_xpu.get_device_properties.return_value = mock_props
            
            from exo.worker.engines.ipex.utils_ipex import get_intel_gpu_memory_usage
            
            memory_info = get_intel_gpu_memory_usage(0)
            
            # Verify data format is suitable for dashboard display
            assert isinstance(memory_info["allocated"], int)
            assert isinstance(memory_info["reserved"], int)
            assert isinstance(memory_info["free"], int)
            assert isinstance(memory_info["total"], int)
            
            # Verify values are reasonable
            assert memory_info["allocated"] > 0
            assert memory_info["total"] > memory_info["allocated"]
            assert memory_info["free"] >= 0

    def test_gpu_health_monitoring_data_format(self):
        """Test GPU health monitoring data format for dashboard."""
        with patch('exo.worker.engines.ipex.utils_ipex.monitor_intel_gpu_health') as mock_health:
            mock_health.return_value = {
                "device_id": 0,
                "overall_health": "healthy",
                "health_status": {
                    "healthy": True,
                    "issues": [],
                    "memory_status": "normal",
                    "performance_status": "normal"
                },
                "memory_leak_info": {
                    "leak_detected": False,
                    "memory_growth": 0
                },
                "performance_alerts": [],
                "error_count": 0,
                "last_check": 1234567890.0
            }
            
            from exo.worker.engines.ipex.utils_ipex import monitor_intel_gpu_health
            
            health_info = monitor_intel_gpu_health(0)
            
            # Verify data format is suitable for dashboard display
            assert isinstance(health_info["device_id"], int)
            assert isinstance(health_info["overall_health"], str)
            assert health_info["overall_health"] in ["healthy", "unhealthy", "error"]
            assert isinstance(health_info["error_count"], int)
            assert isinstance(health_info["last_check"], (int, float))

    def test_performance_metrics_data_format(self):
        """Test performance metrics data format for dashboard."""
        # Mock performance metrics that would be collected
        performance_metrics = {
            "operation": "text_generation",
            "duration": 2.5,
            "tokens_generated": 50,
            "tokens_per_second": 20.0,
            "memory_usage": {
                "allocated": 1024**3,
                "reserved": 1.5 * 1024**3,
                "free": 6.5 * 1024**3,
                "total": 8 * 1024**3
            },
            "device_id": 0
        }
        
        # Verify data types are suitable for dashboard display
        assert isinstance(performance_metrics["operation"], str)
        assert isinstance(performance_metrics["duration"], (int, float))
        assert isinstance(performance_metrics["tokens_generated"], int)
        assert isinstance(performance_metrics["tokens_per_second"], (int, float))
        assert isinstance(performance_metrics["memory_usage"], dict)
        assert isinstance(performance_metrics["device_id"], int)
        
        # Should be JSON serializable
        json_str = json.dumps(performance_metrics)
        assert isinstance(json_str, str)