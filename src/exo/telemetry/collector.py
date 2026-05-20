"""Background telemetry collector for per-node GPU and network metrics.

Runs as an asyncio task at ~1 Hz, collecting GPU metrics via intel_gpu_top
(with sysfs fallback) and network metrics from /proc/net/dev. Reports
NodeTelemetry to the aggregator via a callback.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from exo.shared.types.common import NodeId
from exo.telemetry.models import GpuMetrics, NetworkMetrics, NodeTelemetry
from exo.telemetry.parsers import (
    IntelGpuTopResult,
    NetDevEntry,
    compute_throughput,
    is_cluster_interface,
    parse_intel_gpu_top,
    parse_proc_net_dev,
)

logger: Final = logging.getLogger(__name__)

_SYSFS_GPU_FREQ_PATH: Final = Path("/sys/class/drm/card0/gt_cur_freq_mhz")
_RAPL_ENERGY_PATH: Final = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
_PROC_NET_DEV_PATH: Final = Path("/proc/net/dev")
_INTEL_GPU_TOP_COMMAND: Final = ("intel_gpu_top", "-J", "-s", "900")
_GPU_READ_TIMEOUT_SECONDS: Final[float] = 2.0
_COLLECTION_INTERVAL_SECONDS: Final[float] = 1.0


class TelemetryCollector:
    """Collects GPU and network telemetry from the local node at ~1 Hz."""

    def __init__(
        self,
        node_id: NodeId,
        callback: Callable[[NodeTelemetry], Awaitable[None]],
    ) -> None:
        self._node_id: Final[NodeId] = node_id
        self._callback: Final[Callable[[NodeTelemetry], Awaitable[None]]] = callback
        self._task: asyncio.Task[None] | None = None
        self._gpu_process: asyncio.subprocess.Process | None = None
        self._gpu_process_failed: bool = False

        # Previous network sample for throughput computation
        self._previous_bytes_sent: int | None = None
        self._previous_bytes_received: int | None = None
        self._previous_network_timestamp: datetime | None = None

        # Previous RAPL energy sample for power computation
        self._previous_energy_uj: int | None = None
        self._previous_energy_timestamp: datetime | None = None

    def start(self) -> None:
        """Start the background telemetry collection loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._collection_loop())

    def stop(self) -> None:
        """Stop the background telemetry collection loop and clean up."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._gpu_process is not None:
            with contextlib.suppress(ProcessLookupError):
                self._gpu_process.kill()
            self._gpu_process = None

    async def _collection_loop(self) -> None:
        """Main collection loop running at ~1 Hz."""
        try:
            await self._start_gpu_process()
        except Exception:
            logger.warning("Failed to start intel_gpu_top, will use sysfs fallback")
            self._gpu_process_failed = True

        while True:
            try:
                now = datetime.now(tz=timezone.utc)
                gpu_metrics = await self._collect_gpu(now)
                network_metrics = await self._collect_network(now)

                telemetry = NodeTelemetry(
                    node_id=self._node_id,
                    gpu=gpu_metrics,
                    network=network_metrics,
                )
                await self._callback(telemetry)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telemetry collection iteration failed")

            await asyncio.sleep(_COLLECTION_INTERVAL_SECONDS)

    async def _start_gpu_process(self) -> None:
        """Spawn the intel_gpu_top subprocess."""
        self._gpu_process = await asyncio.create_subprocess_exec(
            *_INTEL_GPU_TOP_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _collect_gpu(self, timestamp: datetime) -> GpuMetrics:
        """Collect GPU metrics from intel_gpu_top or sysfs fallback."""
        power_watts = await self._read_rapl_power(timestamp)

        if not self._gpu_process_failed and self._gpu_process is not None:
            result = await self._read_gpu_top_line()
            if result is not None:
                return GpuMetrics(
                    node_id=self._node_id,
                    timestamp=timestamp,
                    frequency_mhz=result.frequency_mhz,
                    utilization_percent=result.utilization_percent,
                    render_busy_percent=result.render_busy_percent,
                    memory_bandwidth_percent=result.memory_bandwidth_percent,
                    power_watts=power_watts,
                    source="intel_gpu_top",
                )

        # Fallback to sysfs
        frequency = await self._read_sysfs_frequency()
        if frequency is not None:
            return GpuMetrics(
                node_id=self._node_id,
                timestamp=timestamp,
                frequency_mhz=frequency,
                power_watts=power_watts,
                source="sysfs",
            )

        # Both unavailable
        return GpuMetrics(
            node_id=self._node_id,
            timestamp=timestamp,
            power_watts=power_watts,
            source="unavailable",
        )

    async def _read_gpu_top_line(self) -> IntelGpuTopResult | None:
        """Read one JSON line from the intel_gpu_top subprocess."""
        if self._gpu_process is None or self._gpu_process.stdout is None:
            self._gpu_process_failed = True
            return None

        # Check if process is still alive
        if self._gpu_process.returncode is not None:
            self._gpu_process_failed = True
            logger.warning(
                "intel_gpu_top exited with code %d, switching to sysfs fallback",
                self._gpu_process.returncode,
            )
            return None

        try:
            line = await asyncio.wait_for(
                self._gpu_process.stdout.readline(),
                timeout=_GPU_READ_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, TimeoutError):
            return None

        if not line:
            self._gpu_process_failed = True
            return None

        return parse_intel_gpu_top(line.decode("utf-8", errors="replace"))

    async def _read_sysfs_frequency(self) -> int | None:
        """Read GPU frequency from sysfs."""
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None, _SYSFS_GPU_FREQ_PATH.read_text
            )
            return int(content.strip())
        except (OSError, ValueError):
            return None

    async def _read_rapl_power(self, timestamp: datetime) -> float | None:
        """Read RAPL package power in watts from energy_uj delta."""
        try:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(None, _RAPL_ENERGY_PATH.read_text)
            energy_uj = int(raw.strip())
        except (OSError, ValueError):
            return None

        power: float | None = None
        if self._previous_energy_uj is not None and self._previous_energy_timestamp is not None:
            dt = (timestamp - self._previous_energy_timestamp).total_seconds()
            if dt > 0:
                delta = energy_uj - self._previous_energy_uj
                if delta < 0:
                    delta += 2**32
                power = delta * 1e-6 / dt

        self._previous_energy_uj = energy_uj
        self._previous_energy_timestamp = timestamp
        return power

    async def _collect_network(self, timestamp: datetime) -> NetworkMetrics:
        """Collect network metrics from /proc/net/dev."""
        interface_name = "unknown"
        bytes_sent = 0
        bytes_received = 0

        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(
                None, _PROC_NET_DEV_PATH.read_text
            )
            entries = parse_proc_net_dev(content)
            entry = self._find_cluster_entry(entries)
            if entry is not None:
                interface_name = entry.interface_name
                bytes_sent = entry.bytes_sent
                bytes_received = entry.bytes_received
        except OSError:
            logger.warning("Failed to read /proc/net/dev")

        # Compute throughput from previous sample
        throughput_sent = 0.0
        throughput_received = 0.0

        if (
            self._previous_bytes_sent is not None
            and self._previous_bytes_received is not None
            and self._previous_network_timestamp is not None
        ):
            time_delta = (
                timestamp - self._previous_network_timestamp
            ).total_seconds()
            throughput_sent = compute_throughput(
                bytes_sent, self._previous_bytes_sent, time_delta
            )
            throughput_received = compute_throughput(
                bytes_received, self._previous_bytes_received, time_delta
            )

        # Store current sample for next iteration
        self._previous_bytes_sent = bytes_sent
        self._previous_bytes_received = bytes_received
        self._previous_network_timestamp = timestamp

        return NetworkMetrics(
            node_id=self._node_id,
            timestamp=timestamp,
            interface_name=interface_name,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            throughput_sent_bytes_per_sec=throughput_sent,
            throughput_received_bytes_per_sec=throughput_received,
        )

    def _find_cluster_entry(
        self, entries: list[NetDevEntry]
    ) -> NetDevEntry | None:
        """Find the cluster interface entry from parsed /proc/net/dev."""
        for entry in entries:
            if is_cluster_interface(entry.interface_name):
                return entry
        return None
