"""
Thunderbolt 4 Topology Discovery

Frozen dataclasses representing the TB4 network topology discovered between
gremlin nodes, plus the async discovery function that probes the TB4 subnet
to determine reachability and classify the topology.

These models are used by the topology discoverer and the tensor-parallel
process group initialization to determine which nodes are reachable over
high-bandwidth TB4 links (40 Gbps per port).

Requirements: 2.1, 2.2, 2.3, 2.4, 2.6
"""

from __future__ import annotations

import asyncio
import fcntl
import ipaddress
import logging
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TB4Peer:
    """A peer node reachable over Thunderbolt 4.

    Each peer represents a single reachable node on the TB4 subnet,
    identified by its IP address and the local TB4 interface used to
    reach it.
    """

    node_ip: str  # TB4 subnet IP of the peer
    interface_name: str  # Local TB4 interface used to reach this peer
    bandwidth_gbps: float  # Measured or theoretical bandwidth (40.0 for TB4)


@dataclass(frozen=True)
class TB4Topology:
    """Discovered Thunderbolt 4 topology for this node.

    Represents the result of probing the TB4 subnet to determine which
    nodes are directly connected via Thunderbolt 4 cables and what
    topology they form (mesh, ring, partial, or unavailable).
    """

    topology_type: Literal["mesh", "ring", "partial", "unavailable"]
    local_interfaces: list[str]  # TB4 interface names on this node
    local_ips: list[str]  # TB4 IPs assigned to this node
    peers: list[TB4Peer]  # Reachable peers
    all_node_ips: dict[str, list[str]]  # node_hostname -> list of TB4 IPs

    @property
    def is_available(self) -> bool:
        """True if at least 2 nodes are reachable over TB4.

        When unavailable, the system should fall back to pipeline
        parallelism over ethernet.
        """
        return self.topology_type != "unavailable"

    @property
    def world_size(self) -> int:
        """Number of nodes in the TB4 group (including self)."""
        return len(self.all_node_ips)


# Default TCP port used for reachability probes (discard protocol).
_PROBE_PORT = 9

# TB4 theoretical bandwidth per port.
_TB4_BANDWIDTH_GBPS = 40.0


def _enumerate_tb4_interfaces() -> list[tuple[str, str]]:
    """Enumerate active Thunderbolt 4 network interfaces on the local node.

    Checks /sys/class/net/ for interfaces matching thunderbolt* pattern,
    then retrieves their IPv4 addresses via ioctl.

    Returns:
        List of (interface_name, ip_address) tuples for active TB4 interfaces.

    Requirements: 2.1
    """
    results: list[tuple[str, str]] = []
    net_dir = Path("/sys/class/net")

    if not net_dir.exists():
        return results

    for iface_path in sorted(net_dir.iterdir()):
        iface_name = iface_path.name
        if not iface_name.startswith("thunderbolt"):
            continue

        # Check if interface is up by reading operstate
        operstate_path = iface_path / "operstate"
        if operstate_path.exists():
            state = operstate_path.read_text().strip()
            if state not in ("up", "unknown"):
                continue

        # Get IPv4 address via ioctl SIOCGIFADDR
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ip_bytes = fcntl.ioctl(
                sock.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack("256s", iface_name.encode("utf-8")[:15]),
            )[20:24]
            ip_addr = socket.inet_ntoa(ip_bytes)
            sock.close()
            if not ip_addr.startswith("127."):
                results.append((iface_name, ip_addr))
        except (OSError, IOError):
            continue

    return results


async def _probe_tcp_reachability(
    ip: str,
    port: int = _PROBE_PORT,
    timeout: float = 2.0,
) -> bool:
    """Probe a single IP for TCP reachability.

    Attempts an async TCP connection to the given IP and port. A successful
    connection OR a connection-refused response both indicate the host is
    reachable (the network path exists). Only timeouts and network-unreachable
    errors indicate the host is not reachable.

    Args:
        ip: Target IP address to probe.
        port: TCP port to connect to (default: 9, discard protocol).
        timeout: Seconds to wait before declaring unreachable.

    Returns:
        True if the host is reachable, False otherwise.

    Requirements: 2.1, 2.2
    """
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except asyncio.TimeoutError:
        return False
    except ConnectionRefusedError:
        # Connection refused means the host IS reachable (TCP RST received)
        return True
    except OSError:
        # Network unreachable, host unreachable, etc.
        return False


def _classify_topology(
    reachable_nodes: set[str],
    reachability_graph: dict[str, set[str]],
) -> Literal["mesh", "ring", "partial", "unavailable"]:
    """Classify the TB4 topology based on the reachability graph.

    Classification rules:
    - "mesh": ALL node pairs are directly connected (complete graph)
    - "ring": Each node has exactly 2 neighbors forming a single cycle
    - "partial": At least 2 nodes reachable but neither mesh nor ring
    - "unavailable": Fewer than 2 nodes reachable

    Args:
        reachable_nodes: Set of all node IPs that are reachable (including local).
        reachability_graph: Adjacency map: node_ip -> set of directly connected node_ips.

    Returns:
        The topology classification string.

    Requirements: 2.2, 2.3, 2.4, 2.6
    """
    node_count = len(reachable_nodes)

    if node_count < 2:
        return "unavailable"

    # Check for mesh: every node connected to every other node
    is_mesh = all(
        len(reachability_graph.get(node, set())) == node_count - 1
        for node in reachable_nodes
    )
    if is_mesh:
        return "mesh"

    # Check for ring: each node has exactly 2 neighbors and forms a single cycle
    if node_count >= 3:
        all_have_two_neighbors = all(
            len(reachability_graph.get(node, set())) == 2
            for node in reachable_nodes
        )
        if all_have_two_neighbors:
            # Verify it forms a single cycle by traversal
            visited: set[str] = set()
            start = next(iter(reachable_nodes))
            current = start
            prev: str | None = None

            for _ in range(node_count):
                visited.add(current)
                neighbors = reachability_graph.get(current, set())
                # Pick the neighbor that isn't the one we came from
                next_nodes = neighbors - {prev} if prev else neighbors
                if not next_nodes:
                    break
                prev = current
                current = next(iter(next_nodes))

            # It's a ring if we visited all nodes and traversal returns to start
            if visited == reachable_nodes and current == start:
                return "ring"

    return "partial"


async def discover_tb4_topology(
    tb4_subnet: str = "10.4.0.0/24",
    expected_nodes: dict[str, list[str]] | None = None,
    probe_timeout_seconds: float = 2.0,
) -> TB4Topology:
    """Discover TB4 topology by probing configured subnet addresses.

    Enumerates local TB4 interfaces, probes the TB4 subnet for reachable
    peers, builds a reachability graph, and classifies the topology.

    When expected_nodes is provided, probes only those specific IPs rather
    than scanning the entire subnet. This is the recommended mode for
    production use and enables testing without actual TB4 hardware.

    Args:
        tb4_subnet: The TB4 subnet to scan (CIDR notation).
        expected_nodes: Optional mapping of hostname -> expected TB4 IPs.
            If provided, only these IPs are probed.
        probe_timeout_seconds: Timeout for each reachability probe.

    Returns:
        TB4Topology describing the discovered network.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.6
    """
    # Step 1: Enumerate local TB4 interfaces
    local_tb4 = _enumerate_tb4_interfaces()
    local_interfaces = [name for name, _ in local_tb4]
    local_ips = [ip for _, ip in local_tb4]

    # Step 2: Determine which IPs to probe
    if expected_nodes is not None:
        # Use expected_nodes mapping — probe all IPs that aren't ours
        all_node_ips = expected_nodes
        target_ips: set[str] = set()
        for ips in expected_nodes.values():
            target_ips.update(ips)
        # Remove our own IPs from probe targets
        target_ips -= set(local_ips)
    else:
        # Scan the subnet (slower, used when no expected_nodes provided)
        network = ipaddress.IPv4Network(tb4_subnet, strict=False)
        target_ips = set()
        for host in network.hosts():
            ip_str = str(host)
            if ip_str not in local_ips:
                target_ips.add(ip_str)
        all_node_ips = {}

    # Step 3: Probe all target IPs concurrently
    probe_tasks = {
        ip: asyncio.create_task(
            _probe_tcp_reachability(ip, timeout=probe_timeout_seconds)
        )
        for ip in target_ips
    }

    reachable_peer_ips: set[str] = set()
    for ip, task in probe_tasks.items():
        try:
            if await task:
                reachable_peer_ips.add(ip)
        except Exception:
            logger.debug(f"Probe failed for {ip}", exc_info=True)

    # Step 4: Build peers list
    # Assign each reachable peer to the first local TB4 interface
    # (in a real multi-port setup, routing determines which interface is used)
    default_interface = local_interfaces[0] if local_interfaces else "thunderbolt0"
    peers: list[TB4Peer] = []
    for peer_ip in sorted(reachable_peer_ips):
        peers.append(
            TB4Peer(
                node_ip=peer_ip,
                interface_name=default_interface,
                bandwidth_gbps=_TB4_BANDWIDTH_GBPS,
            )
        )

    # Step 5: Build reachability graph for topology classification
    # The graph includes all reachable nodes (local + peers)
    # For classification, we treat each unique node (by hostname or IP group) as a vertex
    if expected_nodes is not None:
        # Group IPs by hostname to determine node-level connectivity
        # A node is "reachable" if ANY of its IPs responded
        reachable_hostnames: set[str] = set()
        local_hostname: str | None = None

        for hostname, ips in expected_nodes.items():
            if set(ips) & set(local_ips):
                local_hostname = hostname
                reachable_hostnames.add(hostname)
            elif set(ips) & reachable_peer_ips:
                reachable_hostnames.add(hostname)

        # If we have local IPs but didn't match a hostname, add "local" as a node
        if local_hostname is None and local_ips:
            local_hostname = "local"
            reachable_hostnames.add("local")
            all_node_ips["local"] = local_ips

        # Build node-level reachability graph
        # In expected_nodes mode, assume all reachable nodes can reach each other
        # (since they're all on the same TB4 subnet and we confirmed reachability)
        reachability_graph: dict[str, set[str]] = {}
        reachable_list = sorted(reachable_hostnames)
        for node in reachable_list:
            reachability_graph[node] = set(reachable_list) - {node}

    else:
        # Without expected_nodes, each reachable IP is treated as a separate node
        all_reachable = set(local_ips) | reachable_peer_ips
        reachable_list_ips = sorted(all_reachable)

        # Build IP-level graph: assume all reachable IPs on the subnet can
        # communicate (they responded to probes from us)
        reachability_graph = {}
        for ip in reachable_list_ips:
            reachability_graph[ip] = set(reachable_list_ips) - {ip}

        # Build all_node_ips from discovered IPs (one "node" per IP without hostname info)
        for ip in reachable_list_ips:
            all_node_ips[ip] = [ip]

        reachable_hostnames = set(reachable_list_ips)

    # Step 6: Classify topology
    topology_type = _classify_topology(reachable_hostnames, reachability_graph)

    # Filter all_node_ips to only include reachable nodes
    if expected_nodes is not None:
        filtered_node_ips: dict[str, list[str]] = {}
        for hostname in reachable_hostnames:
            if hostname in all_node_ips:
                filtered_node_ips[hostname] = all_node_ips[hostname]
        all_node_ips = filtered_node_ips

    return TB4Topology(
        topology_type=topology_type,
        local_interfaces=local_interfaces,
        local_ips=local_ips,
        peers=peers,
        all_node_ips=all_node_ips,
    )


def select_tb4_interface(topology: TB4Topology) -> str | None:
    """Select the best TB4 interface for GLOO_SOCKET_IFNAME.

    For mesh topology: any TB4 interface works (Gloo handles routing).
    For ring/partial topology: select interface connected to the most peers.

    Returns None if TB4 is unavailable.

    Requirements: 2.1, 2.2
    """
    if not topology.is_available:
        return None

    if not topology.local_interfaces:
        return None

    if topology.topology_type == "mesh":
        # For mesh, any interface works — Gloo handles routing
        return topology.local_interfaces[0]

    # For ring and partial topologies, pick the interface connected to the most peers
    interface_peer_count: dict[str, int] = {}
    for peer in topology.peers:
        interface_peer_count[peer.interface_name] = (
            interface_peer_count.get(peer.interface_name, 0) + 1
        )

    if not interface_peer_count:
        # No peers have interface info, fall back to first local interface
        return topology.local_interfaces[0]

    # Return the interface with the highest peer count
    return max(interface_peer_count, key=lambda iface: interface_peer_count[iface])
