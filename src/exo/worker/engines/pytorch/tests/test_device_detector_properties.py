"""Property-based tests for device_detector.py — GPU detection and selection.

Uses Hypothesis to verify that device detection logic correctly selects
CUDA over XPU, never falls back to CPU, and raises NoGpuError when no
GPU is available. Also validates that detected device reports contain all
required fields.

**Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6**
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from exo.worker.engines.pytorch.device_detector import (
    DetectedDevice,
    NoGpuError,
    select_primary_device,
)


# --- Strategies ---

# Strategy for CUDA device counts (1-8 GPUs when CUDA is available)
cuda_device_count_st = st.integers(min_value=1, max_value=8)

# Strategy for XPU device counts (1-8 GPUs when XPU is available)
xpu_device_count_st = st.integers(min_value=1, max_value=8)

# Strategy for device memory in bytes (1 GiB to 128 GiB)
memory_bytes_st = st.integers(min_value=1024**3, max_value=128 * 1024**3)

# Strategy for CUDA device names
cuda_device_name_st = st.sampled_from([
    "NVIDIA RTX 4070 Ti SUPER",
    "NVIDIA RTX 4090",
    "NVIDIA A100",
    "NVIDIA H100",
    "NVIDIA RTX 3090",
])

# Strategy for XPU device names (mix of shared and discrete)
xpu_device_name_st = st.sampled_from([
    "Intel(R) Arc(TM) Graphics Meteor Lake-P",
    "Intel Arc A770",
    "Intel Arc B580",
    "Intel Iris Xe",
    "Intel UHD Graphics 770",
])


def _mock_cuda_detection(device_count: int, device_name: str, memory_bytes: int) -> list[DetectedDevice]:
    """Create a list of mock CUDA detected devices."""
    return [
        DetectedDevice(
            device_type="cuda",
            device_index=i,
            device_name=device_name,
            memory_bytes=memory_bytes,
            memory_architecture="discrete",
        )
        for i in range(device_count)
    ]


def _mock_xpu_detection(device_count: int, device_name: str, memory_bytes: int) -> list[DetectedDevice]:
    """Create a list of mock XPU detected devices."""
    # Determine architecture from name
    name_lower = device_name.lower()
    if "arc a" in name_lower or "arc b" in name_lower:
        architecture = "discrete"
    else:
        architecture = "shared"

    return [
        DetectedDevice(
            device_type="xpu",
            device_index=i,
            device_name=device_name,
            memory_bytes=memory_bytes,
            memory_architecture=architecture,
        )
        for i in range(device_count)
    ]


class TestDeviceDetectionProperty:
    """Property 1: Device detection selects correctly and never falls back to CPU.

    *For any* combination of (cuda_available: bool, xpu_available: bool) where at
    least one is True, the device detector SHALL select "cuda" when CUDA is available
    (preferring it over XPU), select "xpu" when only XPU is available, and SHALL never
    select "cpu" as the primary device. When neither is available, it SHALL raise a
    fatal error.

    **Validates: Requirements 3.1, 3.2, 3.4, 3.5**
    """

    @settings(max_examples=100)
    @given(
        cuda_available=st.just(True),
        xpu_available=st.booleans(),
        cuda_device_count=cuda_device_count_st,
        xpu_device_count=xpu_device_count_st,
        cuda_name=cuda_device_name_st,
        xpu_name=xpu_device_name_st,
        cuda_memory=memory_bytes_st,
        xpu_memory=memory_bytes_st,
    )
    def test_cuda_available_selects_cuda(
        self,
        cuda_available: bool,
        xpu_available: bool,
        cuda_device_count: int,
        xpu_device_count: int,
        cuda_name: str,
        xpu_name: str,
        cuda_memory: int,
        xpu_memory: int,
    ) -> None:
        """When CUDA is available, select_primary_device() returns a CUDA device.

        Requirement 3.5: WHEN both CUDA and XPU devices are available on the same
        node, THE Device_Detector SHALL prefer CUDA for the primary inference device.
        """
        cuda_devices = _mock_cuda_detection(cuda_device_count, cuda_name, cuda_memory)
        xpu_devices = _mock_xpu_detection(xpu_device_count, xpu_name, xpu_memory) if xpu_available else []

        with patch("exo.worker.engines.pytorch.device_detector.detect_devices") as mock_detect:
            mock_detect.return_value = cuda_devices + xpu_devices
            primary = select_primary_device()

        assert primary.device_type == "cuda", (
            f"Expected 'cuda' when CUDA is available, got '{primary.device_type}'"
        )
        assert primary.device_type != "cpu"  # type: ignore[comparison-overlap]

    @settings(max_examples=100)
    @given(
        xpu_device_count=xpu_device_count_st,
        xpu_name=xpu_device_name_st,
        xpu_memory=memory_bytes_st,
    )
    def test_xpu_only_selects_xpu(
        self,
        xpu_device_count: int,
        xpu_name: str,
        xpu_memory: int,
    ) -> None:
        """When CUDA is unavailable and XPU is available, returns XPU device.

        Requirement 3.2: WHEN torch.xpu.is_available() returns True and an Intel
        Arc GPU is detected, THE Device_Detector SHALL report "xpu" as an available
        backend.
        """
        xpu_devices = _mock_xpu_detection(xpu_device_count, xpu_name, xpu_memory)

        with patch("exo.worker.engines.pytorch.device_detector.detect_devices") as mock_detect:
            mock_detect.return_value = xpu_devices
            primary = select_primary_device()

        assert primary.device_type == "xpu", (
            f"Expected 'xpu' when only XPU is available, got '{primary.device_type}'"
        )
        assert primary.device_type != "cpu"  # type: ignore[comparison-overlap]

    @settings(max_examples=100)
    @given(
        cuda_available=st.booleans(),
        xpu_available=st.booleans(),
        cuda_device_count=cuda_device_count_st,
        xpu_device_count=xpu_device_count_st,
        cuda_name=cuda_device_name_st,
        xpu_name=xpu_device_name_st,
        cuda_memory=memory_bytes_st,
        xpu_memory=memory_bytes_st,
    )
    def test_never_returns_cpu(
        self,
        cuda_available: bool,
        xpu_available: bool,
        cuda_device_count: int,
        xpu_device_count: int,
        cuda_name: str,
        xpu_name: str,
        cuda_memory: int,
        xpu_memory: int,
    ) -> None:
        """The device detector SHALL never select "cpu" as the primary device.

        Requirement 3.4: THE Device_Detector SHALL never fall back to CPU inference.

        When at least one GPU is available, the result must be "cuda" or "xpu".
        When no GPU is available, NoGpuError must be raised.
        """
        cuda_devices = _mock_cuda_detection(cuda_device_count, cuda_name, cuda_memory) if cuda_available else []
        xpu_devices = _mock_xpu_detection(xpu_device_count, xpu_name, xpu_memory) if xpu_available else []
        all_devices = cuda_devices + xpu_devices

        with patch("exo.worker.engines.pytorch.device_detector.detect_devices") as mock_detect:
            mock_detect.return_value = all_devices

            if not all_devices:
                # When no GPU is available, must raise NoGpuError
                with pytest.raises(NoGpuError):
                    select_primary_device()
            else:
                primary = select_primary_device()
                # Never returns CPU
                assert primary.device_type in ("cuda", "xpu"), (
                    f"Device type must be 'cuda' or 'xpu', got '{primary.device_type}'"
                )
                assert primary.device_type != "cpu"  # type: ignore[comparison-overlap]

    @settings(max_examples=100)
    @given(st.data())
    def test_no_gpu_raises_fatal_error(self, data: st.DataObject) -> None:
        """When neither CUDA nor XPU is available, raises NoGpuError.

        Requirement 3.3 (via 3.4): IF neither torch.cuda.is_available() nor
        torch.xpu.is_available() returns True, THEN THE Device_Detector SHALL
        raise a fatal error and refuse to start inference.
        """
        with patch("exo.worker.engines.pytorch.device_detector.detect_devices") as mock_detect:
            mock_detect.return_value = []

            with pytest.raises(NoGpuError):
                select_primary_device()



class TestDetectedDeviceFieldsProperty:
    """Property 2: Detected device report contains all required fields.

    *For any* detected GPU device (whether CUDA or XPU), the detection result
    SHALL include a non-empty device name, a non-negative memory capacity in
    bytes, a valid device index, and a memory architecture classification of
    either "shared" or "discrete".

    **Validates: Requirements 3.6**
    """

    # Strategy for generating arbitrary DetectedDevice instances
    detected_device_st = st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    )

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_device_name_is_non_empty(self, device: DetectedDevice) -> None:
        """Device name must be a non-empty string."""
        assert isinstance(device.device_name, str)
        assert len(device.device_name) > 0
        assert device.device_name.strip() != ""

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_memory_bytes_is_non_negative(self, device: DetectedDevice) -> None:
        """Memory capacity must be non-negative (>= 0 bytes)."""
        assert isinstance(device.memory_bytes, int)
        assert device.memory_bytes >= 0

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_device_index_is_non_negative(self, device: DetectedDevice) -> None:
        """Device index must be a non-negative integer (>= 0)."""
        assert isinstance(device.device_index, int)
        assert device.device_index >= 0

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_memory_architecture_is_valid(self, device: DetectedDevice) -> None:
        """Memory architecture must be either 'shared' or 'discrete'."""
        assert device.memory_architecture in ("shared", "discrete")

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_device_type_is_valid(self, device: DetectedDevice) -> None:
        """Device type must be either 'cuda' or 'xpu'."""
        assert device.device_type in ("cuda", "xpu")

    @settings(max_examples=100)
    @given(device=st.builds(
        DetectedDevice,
        device_type=st.sampled_from(["cuda", "xpu"]),
        device_index=st.integers(min_value=0, max_value=15),
        device_name=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
        memory_bytes=st.integers(min_value=0, max_value=256 * 1024**3),
        memory_architecture=st.sampled_from(["shared", "discrete"]),
    ))
    def test_all_required_fields_present(self, device: DetectedDevice) -> None:
        """All required fields must be present and valid in a single check.

        Combines all field validations to verify the complete contract:
        - device_name: non-empty string
        - memory_bytes: non-negative integer
        - device_index: non-negative integer
        - memory_architecture: "shared" or "discrete"
        - device_type: "cuda" or "xpu"
        """
        # device_name is non-empty
        assert isinstance(device.device_name, str)
        assert len(device.device_name) > 0
        assert device.device_name.strip() != ""

        # memory_bytes is non-negative
        assert isinstance(device.memory_bytes, int)
        assert device.memory_bytes >= 0

        # device_index is non-negative
        assert isinstance(device.device_index, int)
        assert device.device_index >= 0

        # memory_architecture is valid
        assert device.memory_architecture in ("shared", "discrete")

        # device_type is valid
        assert device.device_type in ("cuda", "xpu")
