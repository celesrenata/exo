"""Parsers for telemetry data sources.

Provides pure functions to parse GPU statistics from `intel_gpu_top -J`,
network interface statistics from `/proc/net/dev`, compute throughput
from consecutive samples, filter cluster interfaces, and detect metric
staleness.
"""

import ipaddress
import json
from datetime import datetime, timedelta
from typing import Any, NamedTuple, cast


class IntelGpuTopResult(NamedTuple):
    """Parsed result from intel_gpu_top JSON output."""

    frequency_mhz: int
    utilization_percent: float
    render_busy_percent: float
    memory_bandwidth_percent: float | None


class NetDevEntry(NamedTuple):
    """A single interface entry parsed from /proc/net/dev."""

    interface_name: str
    bytes_received: int
    bytes_sent: int


def parse_intel_gpu_top(json_output: str) -> IntelGpuTopResult | None:
    """Parse the JSON output from `intel_gpu_top -J`.

    Expected JSON structure:
        - "engines": dict mapping engine names to objects with a "busy" float
        - "frequency": dict with "actual" (int MHz) and "requested" (int MHz)
        - "rc6": dict with "value" (float, idle percentage)

    The render engine is identified by a name containing "Render/3D".
    Utilization is computed as 100 - rc6 idle percentage.

    Returns an IntelGpuTopResult or None if parsing fails.
    """
    try:
        data: dict[str, Any] = json.loads(json_output)  # pyright: ignore[reportAny]
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None

    # Extract frequency
    frequency: Any = data.get("frequency")
    if not isinstance(frequency, dict):
        return None
    freq_dict = cast(dict[str, Any], frequency)
    actual_freq: Any = freq_dict.get("actual")
    if isinstance(actual_freq, float):
        frequency_mhz = int(actual_freq)
    elif isinstance(actual_freq, int):
        frequency_mhz = actual_freq
    else:
        return None

    # Extract rc6 (idle percentage) -> utilization = 100 - rc6
    rc6: Any = data.get("rc6")
    if not isinstance(rc6, dict):
        return None
    rc6_dict = cast(dict[str, Any], rc6)
    rc6_value: Any = rc6_dict.get("value")
    if not isinstance(rc6_value, (int, float)):
        return None
    utilization_percent: float = 100.0 - float(rc6_value)

    # Find render engine (typically "Render/3D" or "Render/3D/0")
    engines: Any = data.get("engines")
    if not isinstance(engines, dict):
        return None
    engines_dict = cast(dict[str, Any], engines)

    render_busy_percent: float = 0.0
    for engine_name, engine_data in engines_dict.items():  # pyright: ignore[reportAny]
        if not isinstance(engine_data, dict):
            continue
        if "Render/3D" in engine_name:
            eng_dict = cast(dict[str, Any], engine_data)
            busy: Any = eng_dict.get("busy")
            if isinstance(busy, (int, float)):
                render_busy_percent = float(busy)
            break

    # Memory bandwidth (optional, look for engine with "memory" or "bandwidth")
    memory_bandwidth_percent: float | None = None
    for engine_name, engine_data in engines_dict.items():  # pyright: ignore[reportAny]
        if not isinstance(engine_data, dict):
            continue
        engine_name_lower = engine_name.lower()
        if "memory" in engine_name_lower or "bandwidth" in engine_name_lower:
            eng_dict = cast(dict[str, Any], engine_data)
            busy = eng_dict.get("busy")
            if isinstance(busy, (int, float)):
                memory_bandwidth_percent = float(busy)
            break

    return IntelGpuTopResult(
        frequency_mhz=frequency_mhz,
        utilization_percent=utilization_percent,
        render_busy_percent=render_busy_percent,
        memory_bandwidth_percent=memory_bandwidth_percent,
    )


def parse_proc_net_dev(content: str) -> list[NetDevEntry]:
    """Parse the content of `/proc/net/dev`.

    First two lines are headers and are skipped.
    Each subsequent line has the format:
        interface_name: rx_bytes rx_packets rx_errs rx_drop rx_fifo rx_frame rx_compressed rx_multicast tx_bytes tx_packets ...

    bytes_received is column 1 (first number after interface name).
    bytes_sent is column 9 (9th number after interface name).

    Returns a list of NetDevEntry for each valid line.
    """
    entries: list[NetDevEntry] = []
    lines = content.splitlines()

    # Skip first two header lines
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Split on ':' to separate interface name from statistics
        colon_index = stripped.find(":")
        if colon_index == -1:
            continue

        interface_name = stripped[:colon_index].strip()
        numbers_str = stripped[colon_index + 1 :].strip()
        number_parts = numbers_str.split()

        # Need at least 9 fields to get tx_bytes (index 8)
        if len(number_parts) < 9:
            continue

        try:
            bytes_received = int(number_parts[0])
            bytes_sent = int(number_parts[8])
        except (ValueError, IndexError):
            continue

        entries.append(
            NetDevEntry(
                interface_name=interface_name,
                bytes_received=bytes_received,
                bytes_sent=bytes_sent,
            )
        )

    return entries


def compute_throughput(
    bytes_current: int,
    bytes_previous: int,
    time_delta_seconds: float,
) -> float:
    """Compute throughput in bytes per second from consecutive samples.

    Returns 0.0 if time_delta_seconds is zero (avoids division by zero).
    Returns 0.0 if bytes_current < bytes_previous (counter wrap detected).
    """
    if time_delta_seconds == 0.0:
        return 0.0
    if bytes_current < bytes_previous:
        return 0.0
    return float(bytes_current - bytes_previous) / time_delta_seconds


def is_cluster_interface(name: str) -> bool:
    """Check if an interface name passes the cluster interface exclusion filter.

    Returns False for:
        - "lo" (loopback)
        - Names starting with "docker"
        - Names starting with "veth"
        - Names starting with "br-"
        - Names starting with "virbr"

    Returns True for all other interface names.
    """
    if name == "lo":
        return False
    if name.startswith("docker"):
        return False
    if name.startswith("veth"):
        return False
    if name.startswith("br-"):
        return False
    return not name.startswith("virbr")


_CLUSTER_SUBNET = ipaddress.IPv4Network("10.1.1.0/24", strict=False)


def find_cluster_interface_by_ip(
    interfaces: dict[str, list[str]],
) -> str | None:
    """Find the first interface with an IP address in the 10.1.1.0/24 subnet.

    Interfaces are first filtered by the exclusion patterns in
    :func:`is_cluster_interface`. Returns the interface name or None
    if no matching interface is found.
    """
    for name, ip_addresses in interfaces.items():
        if not is_cluster_interface(name):
            continue
        for ip_string in ip_addresses:
            try:
                ip_address = ipaddress.IPv4Address(ip_string)
            except (ipaddress.AddressValueError, ValueError):
                continue
            if ip_address in _CLUSTER_SUBNET:
                return name
    return None


STALENESS_THRESHOLD_SECONDS: float = 3.0


def is_metric_stale(metric_timestamp: datetime, current_time: datetime) -> bool:
    """Determine if a metric timestamp is stale compared to the current time.

    A metric is considered stale if the elapsed time between the metric's
    timestamp and the current time exceeds the staleness threshold (3 seconds).

    Returns True if (current_time - metric_timestamp) > 3 seconds, False otherwise.
    """
    return (current_time - metric_timestamp) > timedelta(seconds=STALENESS_THRESHOLD_SECONDS)
