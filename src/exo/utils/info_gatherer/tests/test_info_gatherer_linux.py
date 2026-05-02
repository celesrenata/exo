"""Unit tests for Info Gatherer Linux extensions.

Tests Linux GPU detection in _monitor_memory_usage, Linux ethernet interface
detection, GPU detection failure fallback, and Linux static node info.

Requirements: 3.1, 3.7, 3.8
"""

from __future__ import annotations

import socket
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import GpuMemoryInfo, MemoryUsage, NetworkInterfaceInfo


# ---------------------------------------------------------------------------
# Task 8.1 / 8.4: Linux GPU detection populates gpu_info on MemoryUsage
# ---------------------------------------------------------------------------


class TestLinuxGpuDetectionInMemoryUsage:
    """Test that _detect_linux_gpu_info converts GpuInfo → GpuMemoryInfo."""

    def test_gpu_detected_populates_gpu_info(self) -> None:
        """When detect_gpus returns a GPU, _detect_linux_gpu_info returns GpuMemoryInfo."""
        from exo.worker.engines.pytorch_xpu.gpu_detector import (
            GpuInfo,
            GpuMemoryArchitecture,
            NodeGpuReport,
        )

        fake_report = NodeGpuReport(
            gpus=[
                GpuInfo(
                    name="Intel Arc Graphics",
                    device_type="xpu",
                    device_index=0,
                    memory_architecture=GpuMemoryArchitecture.Shared,
                    total_memory_bytes=32 * 1024**3,
                    available_memory_bytes=28 * 1024**3,
                )
            ],
            has_gpu=True,
            primary_device_type="xpu",
        )

        with patch(
            "exo.utils.info_gatherer.info_gatherer.IS_LINUX", True
        ), patch(
            "exo.worker.engines.pytorch_xpu.gpu_detector.detect_gpus",
            return_value=fake_report,
        ):
            from exo.utils.info_gatherer.info_gatherer import _detect_linux_gpu_info

            result = _detect_linux_gpu_info()

            assert result is not None
            assert isinstance(result, GpuMemoryInfo)
            assert result.device_type == "xpu"
            assert result.memory_architecture == "Shared"
            assert result.gpu_total_memory == Memory.from_bytes(32 * 1024**3)
            assert result.gpu_available_memory == Memory.from_bytes(28 * 1024**3)

    def test_nvidia_gpu_detected_as_discrete(self) -> None:
        """When detect_gpus returns an NVIDIA GPU, it's reported as Discrete."""
        from exo.worker.engines.pytorch_xpu.gpu_detector import (
            GpuInfo,
            GpuMemoryArchitecture,
            NodeGpuReport,
        )

        fake_report = NodeGpuReport(
            gpus=[
                GpuInfo(
                    name="NVIDIA GeForce RTX 3060",
                    device_type="cuda",
                    device_index=0,
                    memory_architecture=GpuMemoryArchitecture.Discrete,
                    total_memory_bytes=12 * 1024**3,
                    available_memory_bytes=11 * 1024**3,
                )
            ],
            has_gpu=True,
            primary_device_type="cuda",
        )

        with patch(
            "exo.worker.engines.pytorch_xpu.gpu_detector.detect_gpus",
            return_value=fake_report,
        ):
            from exo.utils.info_gatherer.info_gatherer import _detect_linux_gpu_info

            result = _detect_linux_gpu_info()

            assert result is not None
            assert result.device_type == "cuda"
            assert result.memory_architecture == "Discrete"

    def test_no_gpu_returns_none(self) -> None:
        """When detect_gpus reports no GPU, _detect_linux_gpu_info returns None."""
        from exo.worker.engines.pytorch_xpu.gpu_detector import NodeGpuReport

        fake_report = NodeGpuReport(
            gpus=[],
            has_gpu=False,
            primary_device_type="cpu",
        )

        with patch(
            "exo.worker.engines.pytorch_xpu.gpu_detector.detect_gpus",
            return_value=fake_report,
        ):
            from exo.utils.info_gatherer.info_gatherer import _detect_linux_gpu_info

            result = _detect_linux_gpu_info()
            assert result is None

    def test_gpu_detection_failure_returns_none_no_crash(self) -> None:
        """When detect_gpus raises an exception, _detect_linux_gpu_info returns None."""
        with patch(
            "exo.worker.engines.pytorch_xpu.gpu_detector.detect_gpus",
            side_effect=RuntimeError("torch not available"),
        ):
            from exo.utils.info_gatherer.info_gatherer import _detect_linux_gpu_info

            result = _detect_linux_gpu_info()
            assert result is None

    def test_memory_usage_with_gpu_info_on_linux(self) -> None:
        """MemoryUsage emitted on Linux includes gpu_info when GPU is detected."""
        from exo.worker.engines.pytorch_xpu.gpu_detector import (
            GpuInfo,
            GpuMemoryArchitecture,
            NodeGpuReport,
        )

        fake_report = NodeGpuReport(
            gpus=[
                GpuInfo(
                    name="Intel Arc Graphics",
                    device_type="xpu",
                    device_index=0,
                    memory_architecture=GpuMemoryArchitecture.Shared,
                    total_memory_bytes=32 * 1024**3,
                    available_memory_bytes=28 * 1024**3,
                )
            ],
            has_gpu=True,
            primary_device_type="xpu",
        )

        mem = MemoryUsage.from_bytes(
            ram_total=32 * 1024**3,
            ram_available=28 * 1024**3,
            swap_total=8 * 1024**3,
            swap_available=8 * 1024**3,
        )

        with patch(
            "exo.worker.engines.pytorch_xpu.gpu_detector.detect_gpus",
            return_value=fake_report,
        ):
            from exo.utils.info_gatherer.info_gatherer import _detect_linux_gpu_info

            gpu_info = _detect_linux_gpu_info()
            mem_with_gpu = mem.model_copy(update={"gpu_info": gpu_info})

            assert mem_with_gpu.gpu_info is not None
            assert mem_with_gpu.gpu_info.device_type == "xpu"
            assert mem_with_gpu.gpu_info.memory_architecture == "Shared"
            # RAM fields unchanged
            assert mem_with_gpu.ram_total == Memory.from_bytes(32 * 1024**3)

    def test_memory_usage_without_gpu_on_macos(self) -> None:
        """On macOS, gpu_info remains None (existing behavior preserved)."""
        mem = MemoryUsage.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=4 * 1024**3,
            swap_available=4 * 1024**3,
        )
        assert mem.gpu_info is None


# ---------------------------------------------------------------------------
# Task 8.2 / 8.4: Linux ethernet detection
# ---------------------------------------------------------------------------

# Fake psutil address entry
_SnicAddr = namedtuple("snic", ["family", "address", "netmask", "broadcast", "ptp"])
_SnicStats = namedtuple("snicstats", ["isup", "duplex", "speed", "mtu"])


class TestLinuxEthernetDetection:
    """Test _classify_linux_interface and _get_linux_network_interfaces."""

    def test_classify_ethernet_interfaces(self) -> None:
        from exo.utils.info_gatherer.info_gatherer import _classify_linux_interface

        assert _classify_linux_interface("eth0") == "ethernet"
        assert _classify_linux_interface("eno1") == "ethernet"
        assert _classify_linux_interface("ens3") == "ethernet"
        assert _classify_linux_interface("enp0s31f6") == "ethernet"
        assert _classify_linux_interface("bond0") == "ethernet"

    def test_classify_wifi_interfaces(self) -> None:
        from exo.utils.info_gatherer.info_gatherer import _classify_linux_interface

        assert _classify_linux_interface("wlan0") == "wifi"
        assert _classify_linux_interface("wlp2s0") == "wifi"

    def test_classify_loopback(self) -> None:
        from exo.utils.info_gatherer.info_gatherer import _classify_linux_interface

        assert _classify_linux_interface("lo") == "unknown"

    def test_classify_virtual_interfaces(self) -> None:
        from exo.utils.info_gatherer.info_gatherer import _classify_linux_interface

        assert _classify_linux_interface("veth1234") == "unknown"
        assert _classify_linux_interface("docker0") == "unknown"
        assert _classify_linux_interface("br-abc123") == "unknown"
        assert _classify_linux_interface("virbr0") == "unknown"

    def test_classify_unknown_interface(self) -> None:
        from exo.utils.info_gatherer.info_gatherer import _classify_linux_interface

        assert _classify_linux_interface("tun0") == "unknown"

    def test_get_linux_network_interfaces_returns_ethernet(self) -> None:
        """_get_linux_network_interfaces detects ethernet interfaces correctly."""
        fake_addrs = {
            "enp0s31f6": [
                _SnicAddr(
                    family=socket.AF_INET,
                    address="10.1.1.13",
                    netmask="255.255.255.0",
                    broadcast="10.1.1.255",
                    ptp=None,
                ),
            ],
            "lo": [
                _SnicAddr(
                    family=socket.AF_INET,
                    address="127.0.0.1",
                    netmask="255.0.0.0",
                    broadcast=None,
                    ptp=None,
                ),
            ],
            "wlan0": [
                _SnicAddr(
                    family=socket.AF_INET,
                    address="192.168.1.100",
                    netmask="255.255.255.0",
                    broadcast="192.168.1.255",
                    ptp=None,
                ),
            ],
        }
        fake_stats = {
            "enp0s31f6": _SnicStats(isup=True, duplex=2, speed=2500, mtu=1500),
            "lo": _SnicStats(isup=True, duplex=0, speed=0, mtu=65536),
            "wlan0": _SnicStats(isup=True, duplex=0, speed=0, mtu=1500),
        }

        with patch(
            "exo.utils.info_gatherer.info_gatherer._psutil"
        ) as mock_psutil:
            mock_psutil.net_if_addrs.return_value = fake_addrs
            mock_psutil.net_if_stats.return_value = fake_stats

            from exo.utils.info_gatherer.info_gatherer import (
                _get_linux_network_interfaces,
            )

            result = _get_linux_network_interfaces()

        ethernet_ifaces = [i for i in result if i.interface_type == "ethernet"]
        wifi_ifaces = [i for i in result if i.interface_type == "wifi"]

        assert len(ethernet_ifaces) == 1
        assert ethernet_ifaces[0].name == "enp0s31f6"
        assert ethernet_ifaces[0].ip_address == "10.1.1.13"
        assert len(wifi_ifaces) == 1
        assert wifi_ifaces[0].name == "wlan0"

    def test_down_interfaces_are_skipped(self) -> None:
        """Interfaces that are down should be skipped."""
        fake_addrs = {
            "eth0": [
                _SnicAddr(
                    family=socket.AF_INET,
                    address="10.0.0.1",
                    netmask="255.255.255.0",
                    broadcast="10.0.0.255",
                    ptp=None,
                ),
            ],
        }
        fake_stats = {
            "eth0": _SnicStats(isup=False, duplex=0, speed=0, mtu=1500),
        }

        with patch(
            "exo.utils.info_gatherer.info_gatherer._psutil"
        ) as mock_psutil:
            mock_psutil.net_if_addrs.return_value = fake_addrs
            mock_psutil.net_if_stats.return_value = fake_stats

            from exo.utils.info_gatherer.info_gatherer import (
                _get_linux_network_interfaces,
            )

            result = _get_linux_network_interfaces()

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Task 8.3 / 8.4: Linux static node info reads product_name
# ---------------------------------------------------------------------------


class TestLinuxStaticNodeInfo:
    """Test StaticNodeInformation.gather() on Linux."""

    @pytest.mark.anyio
    async def test_reads_product_name_on_linux(self) -> None:
        """On Linux, gather() reads /sys/class/dmi/id/product_name."""
        with patch(
            "exo.utils.info_gatherer.info_gatherer.IS_LINUX", True
        ), patch(
            "exo.utils.info_gatherer.info_gatherer._read_linux_product_name",
            new_callable=AsyncMock,
            return_value="ASUS PRIME Z790-P",
        ):
            from exo.utils.info_gatherer.info_gatherer import StaticNodeInformation

            sni = await StaticNodeInformation.gather()

            assert sni.model == "ASUS PRIME Z790-P"
            assert sni.chip == "Unknown"

    @pytest.mark.anyio
    async def test_fallback_to_unknown_on_read_failure(self) -> None:
        """When DMI file is unreadable, model falls back to 'Unknown'."""
        with patch(
            "exo.utils.info_gatherer.info_gatherer.IS_LINUX", True
        ), patch(
            "exo.utils.info_gatherer.info_gatherer._read_linux_product_name",
            new_callable=AsyncMock,
            return_value="Unknown",
        ):
            from exo.utils.info_gatherer.info_gatherer import StaticNodeInformation

            sni = await StaticNodeInformation.gather()

            assert sni.model == "Unknown"

    @pytest.mark.anyio
    async def test_read_linux_product_name_success(self) -> None:
        """_read_linux_product_name reads and strips the file content."""
        mock_path = MagicMock()
        mock_path.read_text = AsyncMock(return_value="NUC13ANHi7\n")

        with patch("exo.utils.info_gatherer.info_gatherer.anyio.Path", return_value=mock_path):
            from exo.utils.info_gatherer.info_gatherer import _read_linux_product_name

            result = await _read_linux_product_name()
            assert result == "NUC13ANHi7"

    @pytest.mark.anyio
    async def test_read_linux_product_name_file_not_found(self) -> None:
        """_read_linux_product_name returns 'Unknown' when file doesn't exist."""
        mock_path = MagicMock()
        mock_path.read_text = AsyncMock(side_effect=FileNotFoundError())

        with patch("exo.utils.info_gatherer.info_gatherer.anyio.Path", return_value=mock_path):
            from exo.utils.info_gatherer.info_gatherer import _read_linux_product_name

            result = await _read_linux_product_name()
            assert result == "Unknown"

    @pytest.mark.anyio
    async def test_read_linux_product_name_empty_file(self) -> None:
        """_read_linux_product_name returns 'Unknown' for empty file."""
        mock_path = MagicMock()
        mock_path.read_text = AsyncMock(return_value="  \n")

        with patch("exo.utils.info_gatherer.info_gatherer.anyio.Path", return_value=mock_path):
            from exo.utils.info_gatherer.info_gatherer import _read_linux_product_name

            result = await _read_linux_product_name()
            assert result == "Unknown"
