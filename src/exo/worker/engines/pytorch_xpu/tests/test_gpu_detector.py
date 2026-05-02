# Feature: distributed-gpu-sharding, Task 2.3: GPU detector edge-case unit tests
"""
Unit tests for GPU detector edge cases.

**Validates: Requirements 3.5, 3.8**

Tests cover:
- No GPU available → CPU-only report with has_gpu=False
- ImportError on torch.cuda → warning logged, CPU-only fallback
- NVIDIA GPU detected → Discrete architecture, cuda device_type
- Intel XPU detected → Shared architecture for integrated, xpu device_type
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_GPU_DETECTOR_PATH = _THIS_DIR.parent / "gpu_detector.py"


def _load_gpu_detector() -> types.ModuleType:
    """Load gpu_detector.py directly from file, avoiding __init__.py."""
    module_name = "gpu_detector_unit_isolated"
    spec = importlib.util.spec_from_file_location(module_name, _GPU_DETECTOR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_gpu_detector()
GpuMemoryArchitecture = _mod.GpuMemoryArchitecture
detect_gpus = _mod.detect_gpus
GpuInfo = _mod.GpuInfo
NodeGpuReport = _mod.NodeGpuReport


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockCudaDeviceProps:
    """Mock for torch.cuda.get_device_properties return value."""

    name: str
    total_mem: int


@dataclass
class MockXpuDeviceProps:
    """Mock for torch.xpu.get_device_properties return value."""

    name: str
    total_memory: int


# ---------------------------------------------------------------------------
# Tests: No GPU available → CPU-only report
# ---------------------------------------------------------------------------


class TestNoGpuAvailable:
    """When no GPU is detected, detect_gpus() returns a CPU-only report.

    **Validates: Requirement 3.5**
    """

    def test_no_gpu_returns_cpu_only_report(self) -> None:
        """No CUDA and no XPU → has_gpu=False, primary_device_type='cpu', empty gpus."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            report = detect_gpus()

        assert report.has_gpu is False
        assert report.primary_device_type == "cpu"
        assert len(report.gpus) == 0

    def test_no_gpu_report_is_node_gpu_report(self) -> None:
        """CPU-only report is a proper NodeGpuReport instance."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            report = detect_gpus()

        assert isinstance(report, NodeGpuReport)


# ---------------------------------------------------------------------------
# Tests: ImportError on torch.cuda → warning logged, CPU-only fallback
# ---------------------------------------------------------------------------


class TestImportErrorFallback:
    """When GPU detection raises an exception, detect_gpus() falls back to CPU-only.

    **Validates: Requirement 3.8**
    """

    def test_cuda_detection_exception_falls_back_to_cpu(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If _detect_cuda_gpus raises, detect_gpus logs a warning and returns CPU-only."""
        mock_torch = MagicMock()
        # Make torch.cuda.is_available() raise to simulate import/detection failure
        mock_torch.cuda.is_available.side_effect = RuntimeError(
            "CUDA not available"
        )
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            caplog.at_level(logging.WARNING),
        ):
            report = detect_gpus()

        assert report.has_gpu is False
        assert report.primary_device_type == "cpu"
        assert len(report.gpus) == 0
        # Verify a warning was logged about the NVIDIA detection failure
        assert any(
            "NVIDIA GPU detection failed" in record.message
            for record in caplog.records
        ), f"Expected warning about NVIDIA detection failure, got: {[r.message for r in caplog.records]}"

    def test_xpu_detection_exception_falls_back_to_cpu(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If _detect_xpu_gpus raises, detect_gpus logs a warning and returns CPU-only."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        # Make xpu detection raise
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.side_effect = RuntimeError(
            "XPU not available"
        )

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            caplog.at_level(logging.WARNING),
        ):
            report = detect_gpus()

        assert report.has_gpu is False
        assert report.primary_device_type == "cpu"
        assert len(report.gpus) == 0
        assert any(
            "Intel XPU GPU detection failed" in record.message
            for record in caplog.records
        )

    def test_both_detections_fail_returns_cpu_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If both CUDA and XPU detection raise, detect_gpus returns CPU-only with warnings."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA broken")
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.side_effect = RuntimeError("XPU broken")

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            caplog.at_level(logging.WARNING),
        ):
            report = detect_gpus()

        assert report.has_gpu is False
        assert report.primary_device_type == "cpu"
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) >= 2, (
            f"Expected at least 2 warnings, got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# Tests: NVIDIA GPU detected → Discrete architecture, cuda device_type
# ---------------------------------------------------------------------------


class TestNvidiaGpuDetection:
    """NVIDIA GPUs are detected via torch.cuda and classified as Discrete.

    **Validates: Requirements 3.5, 3.8**
    """

    def test_single_nvidia_gpu_detected(self) -> None:
        """A single NVIDIA GPU → has_gpu=True, cuda device_type, Discrete architecture."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = MockCudaDeviceProps(
            name="NVIDIA GeForce RTX 3060",
            total_mem=12 * 1024**3,  # 12 GiB
        )
        mock_torch.cuda.memory_allocated.return_value = 1 * 1024**3  # 1 GiB used
        # No XPU
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            report = detect_gpus()

        assert report.has_gpu is True
        assert report.primary_device_type == "cuda"
        assert len(report.gpus) == 1

        gpu = report.gpus[0]
        assert gpu.device_type == "cuda"
        assert gpu.name == "NVIDIA GeForce RTX 3060"
        assert gpu.device_index == 0
        assert gpu.memory_architecture == GpuMemoryArchitecture.Discrete
        assert gpu.total_memory_bytes == 12 * 1024**3
        assert gpu.available_memory_bytes == 11 * 1024**3

    def test_nvidia_gpu_always_discrete(self) -> None:
        """All NVIDIA GPUs are classified as Discrete, regardless of name."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = MockCudaDeviceProps(
            name="NVIDIA A100-SXM4-80GB",
            total_mem=80 * 1024**3,
        )
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            report = detect_gpus()

        assert report.gpus[0].memory_architecture == GpuMemoryArchitecture.Discrete
        assert report.gpus[0].device_type == "cuda"


# ---------------------------------------------------------------------------
# Tests: Intel XPU detected → Shared architecture for integrated, xpu device_type
# ---------------------------------------------------------------------------


class TestIntelXpuDetection:
    """Intel XPU GPUs are detected via torch.xpu and classified by name keywords.

    **Validates: Requirements 3.5, 3.8**
    """

    def test_intel_integrated_gpu_classified_shared(self) -> None:
        """Intel Iris Xe (integrated) → Shared architecture, xpu device_type, system RAM as total."""
        system_ram = 32 * 1024**3  # 32 GiB
        system_available = 24 * 1024**3  # 24 GiB available

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        xpu_mock = MagicMock()
        xpu_mock.is_available.return_value = True
        xpu_mock.device_count.return_value = 1
        xpu_mock.get_device_properties.return_value = MockXpuDeviceProps(
            name="Intel Iris Xe Graphics",
            total_memory=2 * 1024**3,  # reported VRAM (ignored for shared)
        )
        xpu_mock.memory_allocated.return_value = 0
        mock_torch.xpu = xpu_mock

        mock_psutil = MagicMock()
        mock_vm = MagicMock()
        mock_vm.total = system_ram
        mock_vm.available = system_available
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict(
            sys.modules, {"torch": mock_torch, "psutil": mock_psutil}
        ):
            report = detect_gpus()

        assert report.has_gpu is True
        assert report.primary_device_type == "xpu"
        assert len(report.gpus) == 1

        gpu = report.gpus[0]
        assert gpu.device_type == "xpu"
        assert gpu.name == "Intel Iris Xe Graphics"
        assert gpu.device_index == 0
        assert gpu.memory_architecture == GpuMemoryArchitecture.Shared
        # Shared GPUs report system RAM as total memory
        assert gpu.total_memory_bytes == system_ram
        assert gpu.available_memory_bytes == system_available

    def test_intel_uhd_classified_shared(self) -> None:
        """Intel UHD Graphics (integrated) → Shared architecture."""
        system_ram = 16 * 1024**3

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        xpu_mock = MagicMock()
        xpu_mock.is_available.return_value = True
        xpu_mock.device_count.return_value = 1
        xpu_mock.get_device_properties.return_value = MockXpuDeviceProps(
            name="Intel UHD Graphics 770",
            total_memory=1 * 1024**3,
        )
        xpu_mock.memory_allocated.return_value = 0
        mock_torch.xpu = xpu_mock

        mock_psutil = MagicMock()
        mock_vm = MagicMock()
        mock_vm.total = system_ram
        mock_vm.available = system_ram // 2
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict(
            sys.modules, {"torch": mock_torch, "psutil": mock_psutil}
        ):
            report = detect_gpus()

        assert report.gpus[0].memory_architecture == GpuMemoryArchitecture.Shared
        assert report.gpus[0].total_memory_bytes == system_ram

    def test_intel_arc_discrete_classified_discrete(self) -> None:
        """Intel Arc A770 (discrete) → Discrete architecture, uses own VRAM."""
        arc_vram = 16 * 1024**3  # 16 GiB

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        xpu_mock = MagicMock()
        xpu_mock.is_available.return_value = True
        xpu_mock.device_count.return_value = 1
        xpu_mock.get_device_properties.return_value = MockXpuDeviceProps(
            name="Intel Arc A770",
            total_memory=arc_vram,
        )
        xpu_mock.memory_allocated.return_value = 2 * 1024**3  # 2 GiB used
        mock_torch.xpu = xpu_mock

        # psutil still needed for the import, but shouldn't be used for discrete
        mock_psutil = MagicMock()
        mock_vm = MagicMock()
        mock_vm.total = 32 * 1024**3
        mock_vm.available = 24 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict(
            sys.modules, {"torch": mock_torch, "psutil": mock_psutil}
        ):
            report = detect_gpus()

        assert report.has_gpu is True
        assert report.primary_device_type == "xpu"

        gpu = report.gpus[0]
        assert gpu.device_type == "xpu"
        assert gpu.name == "Intel Arc A770"
        assert gpu.memory_architecture == GpuMemoryArchitecture.Discrete
        # Discrete GPUs report their own VRAM
        assert gpu.total_memory_bytes == arc_vram
        assert gpu.available_memory_bytes == arc_vram - 2 * 1024**3
