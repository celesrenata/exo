"""Transport layer detection and configuration for RDMA, LACP, and Ethernet.

Provides backend selection logic for distributed communication based on
device types, and transport detection/configuration for inter-node
communication.

Detects available network transports (RDMA over Thunderbolt 4, LACP-bonded
Ethernet, plain Ethernet) and configures the Gloo distributed backend to
use the appropriate network interface.

Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Literal

import psutil

logger = logging.getLogger(__name__)

# Bandwidth estimates for each transport type
_BANDWIDTH_GBPS: dict[str, float] = {
    "rdma": 40.0,
    "lacp": 5.0,
    "ethernet": 2.5,
}


@dataclass(frozen=True)
class TransportInfo:
    """Information about the detected and active network transport.

    Attributes:
        transport_type: The active transport type.
        interface_name: The network interface name (e.g., "bond0", "enp2s0").
        bind_address: The IP address bound to the interface.
        bandwidth_gbps: Estimated bandwidth in Gbps.
    """

    transport_type: Literal["rdma", "ethernet", "lacp"]
    interface_name: str
    bind_address: str
    bandwidth_gbps: float


@dataclass(frozen=True)
class TransportConfig:
    """Network transport configuration for distributed communication.

    Attributes:
        transport_type: The transport type to use.
        interface_name: The network interface name (e.g., "bond0", "enp2s0").
        bind_address: The IP address to bind to.
        mtu: MTU size (default 1500).
        lacp_hash_policy: Hash policy for LACP transport (default "layer3+4").
    """

    transport_type: Literal["rdma", "ethernet", "lacp"]
    interface_name: str
    bind_address: str
    mtu: int = 1500
    lacp_hash_policy: str = "layer3+4"


def _get_interface_ip(interface_name: str) -> str | None:
    """Get the IPv4 address for a network interface.

    Args:
        interface_name: The name of the network interface.

    Returns:
        The IPv4 address string, or None if not found.
    """
    addrs = psutil.net_if_addrs()
    if interface_name not in addrs:
        return None
    for addr in addrs[interface_name]:
        if addr.family == socket.AF_INET:
            return addr.address
    return None


def _detect_rdma_interface() -> tuple[str, str] | None:
    """Detect an RDMA-capable interface (Thunderbolt 4).

    Checks for InfiniBand/RDMA interfaces via /sys/class/infiniband/ and
    for thunderbolt network interfaces in psutil.

    Returns:
        Tuple of (interface_name, ip_address) if found, None otherwise.
    """
    # Check for InfiniBand/RDMA devices
    if os.path.exists("/sys/class/infiniband/"):
        try:
            entries = os.listdir("/sys/class/infiniband/")
            if entries:
                # Look for a corresponding network interface
                addrs = psutil.net_if_addrs()
                for iface_name in addrs:
                    if "thunder" in iface_name.lower() or "rdma" in iface_name.lower():
                        ip = _get_interface_ip(iface_name)
                        if ip:
                            return (iface_name, ip)
        except OSError:
            pass

    # Check for thunderbolt interfaces directly
    addrs = psutil.net_if_addrs()
    for iface_name in addrs:
        if "thunder" in iface_name.lower() or "tb" in iface_name.lower():
            ip = _get_interface_ip(iface_name)
            if ip:
                return (iface_name, ip)

    return None


def _detect_lacp_interface() -> tuple[str, str] | None:
    """Detect an LACP-bonded interface.

    Checks for bond interfaces (e.g., bond0) in the system's network
    interfaces.

    Returns:
        Tuple of (interface_name, ip_address) if found, None otherwise.
    """
    addrs = psutil.net_if_addrs()
    for iface_name in sorted(addrs.keys()):
        if iface_name.startswith("bond"):
            ip = _get_interface_ip(iface_name)
            if ip:
                return (iface_name, ip)
    return None


def _detect_ethernet_interface() -> tuple[str, str] | None:
    """Detect the primary Ethernet interface.

    Looks for standard Ethernet interfaces (enp*, eth*, eno*) with an
    IPv4 address, excluding loopback and virtual interfaces.

    Returns:
        Tuple of (interface_name, ip_address) if found, None otherwise.
    """
    addrs = psutil.net_if_addrs()
    # Prefer interfaces with common Ethernet naming patterns
    ethernet_prefixes = ("enp", "eth", "eno", "ens")

    for iface_name in sorted(addrs.keys()):
        if any(iface_name.startswith(prefix) for prefix in ethernet_prefixes):
            ip = _get_interface_ip(iface_name)
            if ip and ip != "127.0.0.1":
                return (iface_name, ip)

    # Fallback: any non-loopback interface with an IPv4 address
    for iface_name in sorted(addrs.keys()):
        if iface_name == "lo":
            continue
        ip = _get_interface_ip(iface_name)
        if ip and ip != "127.0.0.1":
            return (iface_name, ip)

    return None


def detect_transport(
    preferred: Literal["rdma", "ethernet", "lacp"],
) -> TransportInfo:
    """Detect available transport and configure the network interface.

    Attempts to use the preferred transport. If the preferred transport is
    unavailable, falls back to Ethernet and logs a warning.

    Sets the GLOO_SOCKET_IFNAME environment variable to the detected
    interface name so that Gloo binds to the correct network interface.

    Reports the active transport type and bandwidth at initialization.

    Args:
        preferred: The preferred transport type to use.

    Returns:
        TransportInfo describing the active transport.

    Raises:
        RuntimeError: If no usable network interface is found at all.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
    """
    detected: tuple[str, str] | None = None
    active_transport: Literal["rdma", "ethernet", "lacp"] = preferred

    if preferred == "rdma":
        detected = _detect_rdma_interface()
        if detected is None:
            logger.warning(
                "RDMA transport requested but unavailable — "
                "falling back to Ethernet"
            )
            active_transport = "ethernet"
            detected = _detect_ethernet_interface()
    elif preferred == "lacp":
        detected = _detect_lacp_interface()
        if detected is None:
            logger.warning(
                "LACP transport requested but no bond interface found — "
                "falling back to Ethernet"
            )
            active_transport = "ethernet"
            detected = _detect_ethernet_interface()
    else:
        # preferred == "ethernet"
        detected = _detect_ethernet_interface()

    if detected is None:
        raise RuntimeError(
            "No usable network interface found for transport. "
            f"Preferred: {preferred}, attempted fallback to ethernet. "
            f"Available interfaces: {list(psutil.net_if_addrs().keys())}"
        )

    interface_name, bind_address = detected
    bandwidth = _BANDWIDTH_GBPS[active_transport]

    # Set GLOO_SOCKET_IFNAME so Gloo binds to the correct interface
    os.environ["GLOO_SOCKET_IFNAME"] = interface_name

    logger.info(
        "Transport configured: type=%s, interface=%s, "
        "bind_address=%s, bandwidth=%.1f Gbps",
        active_transport,
        interface_name,
        bind_address,
        bandwidth,
    )

    return TransportInfo(
        transport_type=active_transport,
        interface_name=interface_name,
        bind_address=bind_address,
        bandwidth_gbps=bandwidth,
    )


def select_backend_for_devices(
    device_types: list[Literal["cuda", "xpu"]],
) -> Literal["gloo", "nccl"]:
    """Select the distributed backend based on participating device types.

    Returns "nccl" if and only if ALL device types are "cuda". Returns "gloo"
    otherwise (i.e., when any device is "xpu" or the list is mixed).

    NCCL is optimized for NVIDIA GPUs but does not support Intel XPU devices.
    Gloo supports both device types and serves as the common denominator for
    mixed or all-XPU clusters.

    Args:
        device_types: Non-empty list of device types participating in the
            distributed group. Each element is either "cuda" or "xpu".

    Returns:
        "nccl" if all devices are CUDA, "gloo" otherwise.

    Raises:
        ValueError: If device_types is empty.

    Requirements: 6.1, 6.2, 6.3
    """
    if not device_types:
        raise ValueError("device_types must be a non-empty list")

    if all(dt == "cuda" for dt in device_types):
        return "nccl"
    return "gloo"
