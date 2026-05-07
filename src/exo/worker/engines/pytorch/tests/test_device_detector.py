"""Unit tests for device_detector.py — GPU detection and selection.

Tests use mocked torch backends to verify detection logic without
requiring actual GPU hardware.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from exo.worker.engines.pytorch.device_detector import (
    DetectedDevice,
    NoGpuError,
    _classify_xpu_memory_architecture,
    detect_devices,
    select_primary_device,
)


class TestDetectedDevice:
    """Tests for the DetectedDevice dataclass."""

    def test_frozen_dataclass(self) -> None:
        device = DetectedDevice(
            device_type="cuda",
            device_index=0,
            device_name="NVIDIA RTX 4070 Ti SUPER",
            memory_bytes=16 * 1024**3,
            memory_architecture="discrete",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            device.device_type = "xpu"  # type: ignore[misc]

    def test_cuda_device_fields(self) -> None:
        device = DetectedDevice(
            device_type="cuda",
            device_index=0,
            device_name="NVIDIA RTX 4070 Ti SUPER",
            memory_bytes=16 * 1024**3,
            memory_architecture="discrete",
        )
        assert device.device_type == "cuda"
        assert device.device_index == 0
        assert device.device_name == "NVIDIA RTX 4070 Ti SUPER"
        assert device.memory_bytes == 16 * 1024**3
        assert device.memory_architecture == "discrete"

    def test_xpu_device_fields(self) -> None:
        device = DetectedDevice(
            device_type="xpu",
            device_index=0,
            device_name="Intel(R) Arc(TM) Graphics",
            memory_bytes=94 * 1024**3,
            memory_architecture="shared",
        )
        assert device.device_type == "xpu"
        assert device.device_index == 0
        assert device.device_name == "Intel(R) Arc(TM) Graphics"
        assert device.memory_bytes == 94 * 1024**3
        assert device.memory_architecture == "shared"


class TestClassifyXpuMemoryArchitecture:
    """Tests for _classify_xpu_memory_architecture."""

    def test_meteor_lake_is_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel(R) Arc(TM) Graphics Meteor Lake-P") == "shared"

    def test_integrated_keyword_is_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Integrated Graphics") == "shared"

    def test_iris_is_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Iris Xe") == "shared"

    def test_uhd_is_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel UHD Graphics 770") == "shared"

    def test_core_ultra_is_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Core Ultra Graphics") == "shared"

    def test_arc_a770_is_discrete(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Arc A770") == "discrete"

    def test_arc_b580_is_discrete(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Arc B580") == "discrete"

    def test_unknown_defaults_to_shared(self) -> None:
        assert _classify_xpu_memory_architecture("Intel Unknown GPU") == "shared"


class TestDetectDevices:
    """Tests for detect_devices()."""

    @patch("exo.worker.engines.pytorch.device_detector.torch", create=True)
    def test_cuda_only(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        props = SimpleNamespace(name="NVIDIA RTX 4070 Ti SUPER", total_memory=16 * 1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch("exo.worker.engines.pytorch.device_detector._detect_cuda_devices") as mock_cuda, patch(
            "exo.worker.engines.pytorch.device_detector._detect_xpu_devices"
        ) as mock_xpu:
            mock_cuda.return_value = [
                DetectedDevice(
                    device_type="cuda",
                    device_index=0,
                    device_name="NVIDIA RTX 4070 Ti SUPER",
                    memory_bytes=16 * 1024**3,
                    memory_architecture="discrete",
                )
            ]
            mock_xpu.return_value = []
            devices = detect_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "cuda"
        assert devices[0].device_name == "NVIDIA RTX 4070 Ti SUPER"

    @patch("exo.worker.engines.pytorch.device_detector._detect_xpu_devices")
    @patch("exo.worker.engines.pytorch.device_detector._detect_cuda_devices")
    def test_xpu_only(self, mock_cuda: MagicMock, mock_xpu: MagicMock) -> None:
        mock_cuda.return_value = []
        mock_xpu.return_value = [
            DetectedDevice(
                device_type="xpu",
                device_index=0,
                device_name="Intel(R) Arc(TM) Graphics",
                memory_bytes=94 * 1024**3,
                memory_architecture="shared",
            )
        ]
        devices = detect_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "xpu"

    @patch("exo.worker.engines.pytorch.device_detector._detect_xpu_devices")
    @patch("exo.worker.engines.pytorch.device_detector._detect_cuda_devices")
    def test_both_cuda_and_xpu(self, mock_cuda: MagicMock, mock_xpu: MagicMock) -> None:
        mock_cuda.return_value = [
            DetectedDevice(
                device_type="cuda",
                device_index=0,
                device_name="NVIDIA RTX 4070 Ti SUPER",
                memory_bytes=16 * 1024**3,
                memory_architecture="discrete",
            )
        ]
        mock_xpu.return_value = [
            DetectedDevice(
                device_type="xpu",
                device_index=0,
                device_name="Intel(R) Arc(TM) Graphics",
                memory_bytes=94 * 1024**3,
                memory_architecture="shared",
            )
        ]
        devices = detect_devices()

        assert len(devices) == 2
        cuda_devices = [d for d in devices if d.device_type == "cuda"]
        xpu_devices = [d for d in devices if d.device_type == "xpu"]
        assert len(cuda_devices) == 1
        assert len(xpu_devices) == 1

    @patch("exo.worker.engines.pytorch.device_detector._detect_xpu_devices")
    @patch("exo.worker.engines.pytorch.device_detector._detect_cuda_devices")
    def test_no_devices(self, mock_cuda: MagicMock, mock_xpu: MagicMock) -> None:
        mock_cuda.return_value = []
        mock_xpu.return_value = []
        devices = detect_devices()

        assert devices == []


class TestSelectPrimaryDevice:
    """Tests for select_primary_device()."""

    @patch("exo.worker.engines.pytorch.device_detector.detect_devices")
    def test_prefers_cuda_over_xpu(self, mock_detect: MagicMock) -> None:
        mock_detect.return_value = [
            DetectedDevice(
                device_type="cuda",
                device_index=0,
                device_name="NVIDIA RTX 4070 Ti SUPER",
                memory_bytes=16 * 1024**3,
                memory_architecture="discrete",
            ),
            DetectedDevice(
                device_type="xpu",
                device_index=0,
                device_name="Intel(R) Arc(TM) Graphics",
                memory_bytes=94 * 1024**3,
                memory_architecture="shared",
            ),
        ]
        primary = select_primary_device()

        assert primary.device_type == "cuda"
        assert primary.device_name == "NVIDIA RTX 4070 Ti SUPER"

    @patch("exo.worker.engines.pytorch.device_detector.detect_devices")
    def test_selects_xpu_when_no_cuda(self, mock_detect: MagicMock) -> None:
        mock_detect.return_value = [
            DetectedDevice(
                device_type="xpu",
                device_index=0,
                device_name="Intel(R) Arc(TM) Graphics",
                memory_bytes=94 * 1024**3,
                memory_architecture="shared",
            ),
        ]
        primary = select_primary_device()

        assert primary.device_type == "xpu"

    @patch("exo.worker.engines.pytorch.device_detector.detect_devices")
    def test_raises_no_gpu_error_when_empty(self, mock_detect: MagicMock) -> None:
        mock_detect.return_value = []

        with pytest.raises(NoGpuError, match="No GPU devices detected"):
            select_primary_device()

    @patch("exo.worker.engines.pytorch.device_detector.detect_devices")
    def test_never_returns_cpu(self, mock_detect: MagicMock) -> None:
        """Verify that select_primary_device never returns a CPU device."""
        mock_detect.return_value = [
            DetectedDevice(
                device_type="xpu",
                device_index=0,
                device_name="Intel(R) Arc(TM) Graphics",
                memory_bytes=94 * 1024**3,
                memory_architecture="shared",
            ),
        ]
        primary = select_primary_device()

        assert primary.device_type in ("cuda", "xpu")
        assert primary.device_type != "cpu"  # type: ignore[comparison-overlap]
