# Feature: distributed-gpu-sharding, Property 4: GPU detection and classification
"""
Property-based tests for GPU detection and classification.

**Validates: Requirements 3.1, 3.2, 3.3, 3.6**

Uses Hypothesis to generate random GPU property combinations, mocking torch.cuda
and torch.xpu to verify correct device_type, non-empty name, correct
memory_architecture classification, and positive total_memory_bytes.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_GPU_DETECTOR_PATH = _THIS_DIR.parent / "gpu_detector.py"


def _load_gpu_detector() -> types.ModuleType:
    """Load gpu_detector.py directly from file, avoiding __init__.py."""
    module_name = "gpu_detector_isolated"
    spec = importlib.util.spec_from_file_location(module_name, _GPU_DETECTOR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_gpu_detector()
GpuMemoryArchitecture = _mod.GpuMemoryArchitecture
_classify_xpu_memory_architecture = _mod._classify_xpu_memory_architecture
detect_gpus = _mod.detect_gpus
GpuInfo = _mod.GpuInfo
NodeGpuReport = _mod.NodeGpuReport

# ---------------------------------------------------------------------------
# Hypothesis strategies for GPU configurations
# ---------------------------------------------------------------------------

# Intel integrated GPU name fragments — these trigger Shared classification
_INTEGRATED_NAMES: list[str] = [
    "Intel Iris Xe Graphics",
    "Intel UHD Graphics 770",
    "Intel Core Ultra Integrated GPU",
    "Intel Iris Plus Graphics 655",
    "Intel UHD Graphics 630",
]

# Intel discrete GPU name fragments — these trigger Discrete classification
_DISCRETE_NAMES: list[str] = [
    "Intel Arc A770",
    "Intel Arc A750",
    "Intel Arc A380",
    "Intel Arc B580",
]

# Names that don't match any keyword — default to Shared
_UNRECOGNIZED_NAMES: list[str] = [
    "Intel Graphics",
    "Intel Xe HPG",
    "Intel DG1",
]

# NVIDIA GPU names
_NVIDIA_NAMES: list[str] = [
    "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA Tesla V100-SXM2-16GB",
    "NVIDIA GeForce GTX 1080 Ti",
]

# Positive memory in bytes (1 MiB to 128 GiB)
_memory_bytes = st.integers(min_value=1024 * 1024, max_value=128 * 1024**3)

# Positive system RAM (4 GiB to 256 GiB)
_system_ram_bytes = st.integers(min_value=4 * 1024**3, max_value=256 * 1024**3)


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


@dataclass
class CudaGpuConfig:
    """Configuration for a single mocked CUDA GPU."""

    name: str
    total_memory: int
    allocated_memory: int


@dataclass
class XpuGpuConfig:
    """Configuration for a single mocked XPU GPU."""

    name: str
    category: Literal["integrated", "discrete", "unrecognized"]
    total_memory: int
    allocated_memory: int


# Strategy: generate a CUDA GPU config
def _cuda_gpu_config() -> st.SearchStrategy[CudaGpuConfig]:
    return st.builds(
        CudaGpuConfig,
        name=st.sampled_from(_NVIDIA_NAMES),
        total_memory=_memory_bytes,
        allocated_memory=st.just(0),
    ).flatmap(
        lambda cfg: st.builds(
            CudaGpuConfig,
            name=st.just(cfg.name),
            total_memory=st.just(cfg.total_memory),
            allocated_memory=st.integers(min_value=0, max_value=cfg.total_memory),
        )
    )


# Strategy: generate an XPU GPU config with a known category
def _xpu_gpu_config() -> st.SearchStrategy[XpuGpuConfig]:
    integrated = st.sampled_from(_INTEGRATED_NAMES).map(
        lambda n: ("integrated", n)
    )
    discrete = st.sampled_from(_DISCRETE_NAMES).map(lambda n: ("discrete", n))
    unrecognized = st.sampled_from(_UNRECOGNIZED_NAMES).map(
        lambda n: ("unrecognized", n)
    )

    return st.one_of(integrated, discrete, unrecognized).flatmap(
        lambda cat_name: _memory_bytes.flatmap(
            lambda total: st.builds(
                XpuGpuConfig,
                name=st.just(cat_name[1]),
                category=st.just(cat_name[0]),
                total_memory=st.just(total),
                allocated_memory=st.integers(min_value=0, max_value=total),
            )
        )
    )


def _setup_cuda_mock(
    mock_torch: MagicMock,
    cuda_configs: list[CudaGpuConfig],
) -> None:
    """Configure mock torch.cuda based on CUDA GPU configs."""
    has_cuda = len(cuda_configs) > 0
    mock_torch.cuda.is_available.return_value = has_cuda
    mock_torch.cuda.device_count.return_value = len(cuda_configs)

    def get_cuda_props(i: int) -> MockCudaDeviceProps:
        cfg = cuda_configs[i]
        return MockCudaDeviceProps(name=cfg.name, total_mem=cfg.total_memory)

    mock_torch.cuda.get_device_properties.side_effect = get_cuda_props

    def cuda_mem_allocated(i: int) -> int:
        return cuda_configs[i].allocated_memory

    mock_torch.cuda.memory_allocated.side_effect = cuda_mem_allocated


def _setup_xpu_mock(
    mock_torch: MagicMock,
    xpu_configs: list[XpuGpuConfig],
) -> None:
    """Configure mock torch.xpu based on XPU GPU configs."""
    has_xpu = len(xpu_configs) > 0

    xpu_mock = MagicMock()
    xpu_mock.is_available.return_value = has_xpu
    xpu_mock.device_count.return_value = len(xpu_configs)

    def get_xpu_props(i: int) -> MockXpuDeviceProps:
        cfg = xpu_configs[i]
        return MockXpuDeviceProps(name=cfg.name, total_memory=cfg.total_memory)

    xpu_mock.get_device_properties.side_effect = get_xpu_props

    def xpu_mem_allocated(i: int) -> int:
        return xpu_configs[i].allocated_memory

    xpu_mock.memory_allocated.side_effect = xpu_mem_allocated

    mock_torch.xpu = xpu_mock


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestGpuClassificationProperty:
    """Property 4: GPU detection and classification.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.6**
    """

    @given(
        cuda_configs=st.lists(_cuda_gpu_config(), min_size=1, max_size=4),
        system_ram=_system_ram_bytes,
    )
    @settings(max_examples=100)
    def test_cuda_gpus_always_classified_discrete(
        self,
        cuda_configs: list[CudaGpuConfig],
        system_ram: int,
    ) -> None:
        """For any NVIDIA GPU detected via torch.cuda, device_type SHALL be
        'cuda', memory_architecture SHALL be Discrete, name SHALL be non-empty,
        and total_memory_bytes SHALL be positive.

        **Validates: Requirements 3.1, 3.6**
        """
        mock_torch = MagicMock()
        _setup_cuda_mock(mock_torch, cuda_configs)
        # No XPU available
        mock_torch.xpu = MagicMock()
        mock_torch.xpu.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            report = detect_gpus()

        assert report.has_gpu is True
        assert report.primary_device_type == "cuda"
        assert len(report.gpus) == len(cuda_configs)

        for gpu, cfg in zip(report.gpus, cuda_configs):
            assert gpu.device_type == "cuda"
            assert gpu.name == cfg.name
            assert gpu.name != ""
            assert gpu.memory_architecture == GpuMemoryArchitecture.Discrete
            assert gpu.total_memory_bytes > 0
            assert gpu.total_memory_bytes == cfg.total_memory

    @given(
        xpu_configs=st.lists(_xpu_gpu_config(), min_size=1, max_size=4),
        system_ram=_system_ram_bytes,
    )
    @settings(max_examples=100)
    def test_xpu_gpus_classified_by_name_keywords(
        self,
        xpu_configs: list[XpuGpuConfig],
        system_ram: int,
    ) -> None:
        """For any Intel XPU GPU, device_type SHALL be 'xpu', name SHALL be
        non-empty, total_memory_bytes SHALL be positive, and memory_architecture
        SHALL match the classification rules:
        - Names containing 'iris', 'uhd', 'integrated', 'core ultra' → Shared
        - Names containing 'arc a', 'arc b' → Discrete
        - Unrecognized names → Shared (safe default)

        **Validates: Requirements 3.2, 3.3, 3.6**
        """
        system_available = system_ram // 2

        mock_torch = MagicMock()
        # No CUDA
        mock_torch.cuda.is_available.return_value = False

        _setup_xpu_mock(mock_torch, xpu_configs)

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
        assert len(report.gpus) == len(xpu_configs)

        for gpu, cfg in zip(report.gpus, xpu_configs):
            assert gpu.device_type == "xpu"
            assert gpu.name == cfg.name
            assert gpu.name != ""
            assert gpu.total_memory_bytes > 0

            if cfg.category == "integrated":
                assert gpu.memory_architecture == GpuMemoryArchitecture.Shared, (
                    f"Integrated GPU '{cfg.name}' should be Shared, "
                    f"got {gpu.memory_architecture}"
                )
                # Shared GPUs report system RAM as total memory
                assert gpu.total_memory_bytes == system_ram
            elif cfg.category == "discrete":
                assert gpu.memory_architecture == GpuMemoryArchitecture.Discrete, (
                    f"Discrete GPU '{cfg.name}' should be Discrete, "
                    f"got {gpu.memory_architecture}"
                )
                # Discrete GPUs report their own VRAM
                assert gpu.total_memory_bytes == cfg.total_memory
            else:
                # Unrecognized defaults to Shared
                assert gpu.memory_architecture == GpuMemoryArchitecture.Shared, (
                    f"Unrecognized GPU '{cfg.name}' should default to Shared, "
                    f"got {gpu.memory_architecture}"
                )
                assert gpu.total_memory_bytes == system_ram

    @given(
        cuda_configs=st.lists(_cuda_gpu_config(), min_size=0, max_size=3),
        xpu_configs=st.lists(_xpu_gpu_config(), min_size=0, max_size=3),
        system_ram=_system_ram_bytes,
    )
    @settings(max_examples=100)
    def test_detect_gpus_report_invariants(
        self,
        cuda_configs: list[CudaGpuConfig],
        xpu_configs: list[XpuGpuConfig],
        system_ram: int,
    ) -> None:
        """For any combination of CUDA and XPU GPUs, the NodeGpuReport SHALL
        have has_gpu=True iff gpus is non-empty, primary_device_type SHALL
        prefer 'cuda' over 'xpu', and all GPUs SHALL have non-empty names
        and positive total_memory_bytes.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.6**
        """
        total_gpus = len(cuda_configs) + len(xpu_configs)
        system_available = system_ram // 2

        mock_torch = MagicMock()
        _setup_cuda_mock(mock_torch, cuda_configs)
        _setup_xpu_mock(mock_torch, xpu_configs)

        mock_psutil = MagicMock()
        mock_vm = MagicMock()
        mock_vm.total = system_ram
        mock_vm.available = system_available
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict(
            sys.modules, {"torch": mock_torch, "psutil": mock_psutil}
        ):
            report = detect_gpus()

        if total_gpus == 0:
            assert report.has_gpu is False
            assert report.primary_device_type == "cpu"
            assert len(report.gpus) == 0
        else:
            assert report.has_gpu is True
            assert len(report.gpus) == total_gpus

            # Primary device type: prefer CUDA over XPU
            if len(cuda_configs) > 0:
                assert report.primary_device_type == "cuda"
            else:
                assert report.primary_device_type == "xpu"

            # All GPUs must have non-empty names and positive memory
            for gpu in report.gpus:
                assert gpu.name != ""
                assert gpu.total_memory_bytes > 0
                assert gpu.device_type in ("cuda", "xpu")
                assert gpu.memory_architecture in (
                    GpuMemoryArchitecture.Shared,
                    GpuMemoryArchitecture.Discrete,
                )

    @given(device_name=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_xpu_classification_always_returns_valid_architecture(
        self,
        device_name: str,
    ) -> None:
        """For any non-empty device name string, _classify_xpu_memory_architecture
        SHALL return either Shared or Discrete — never raise an exception.

        **Validates: Requirements 3.3, 3.6**
        """
        result = _classify_xpu_memory_architecture(device_name)
        assert result in (
            GpuMemoryArchitecture.Shared,
            GpuMemoryArchitecture.Discrete,
        )
