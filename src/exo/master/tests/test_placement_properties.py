"""Property-based tests for placement module extensions.

Tests Properties 6 and 7 from the distributed-gpu-sharding design document.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.master.placement_utils import get_pytorch_ring_hosts_by_node
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.topology import Connection, Cycle, SocketConnection


# --- Strategies ---


def _ethernet_ip(index: int) -> str:
    """Generate a deterministic ethernet IP from an index."""
    return f"10.1.1.{(index % 254) + 1}"


@st.composite
def topology_with_ethernet(draw: st.DrawFn):
    """Generate a random fully-connected topology with ethernet interfaces.

    Returns (cycle, topology, node_network, ephemeral_port, node_ids).
    """
    num_nodes = draw(st.integers(min_value=2, max_value=6))
    ephemeral_port = draw(st.integers(min_value=49153, max_value=65535))

    node_ids = [NodeId(f"node-{i}") for i in range(num_nodes)]

    topology = Topology()
    for nid in node_ids:
        topology.add_node(nid)

    # Build fully-connected directed graph with socket connections
    # Each edge from node_i to node_j uses node_j's ethernet IP
    node_network: dict[NodeId, NodeNetworkInfo] = {}
    for i, nid in enumerate(node_ids):
        ip = _ethernet_ip(i)
        node_network[nid] = NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name=f"eth0",
                    ip_address=ip,
                    interface_type="ethernet",
                ),
            ]
        )

    for i, src in enumerate(node_ids):
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            dst_ip = _ethernet_ip(j)
            conn = Connection(
                source=src,
                sink=dst,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(address=f"/ip4/{dst_ip}/tcp/8000")
                ),
            )
            topology.add_connection(conn)

    cycle = Cycle(node_ids=node_ids)

    return cycle, topology, node_network, ephemeral_port, node_ids


@st.composite
def topology_with_mixed_interfaces(draw: st.DrawFn):
    """Generate topology where nodes have both ethernet and wifi interfaces.

    Ethernet should always be prioritized.
    """
    num_nodes = draw(st.integers(min_value=2, max_value=5))
    ephemeral_port = draw(st.integers(min_value=49153, max_value=65535))

    node_ids = [NodeId(f"node-{i}") for i in range(num_nodes)]

    topology = Topology()
    for nid in node_ids:
        topology.add_node(nid)

    node_network: dict[NodeId, NodeNetworkInfo] = {}
    for i, nid in enumerate(node_ids):
        eth_ip = _ethernet_ip(i)
        wifi_ip = f"192.168.1.{(i % 254) + 1}"
        node_network[nid] = NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="wlan0",
                    ip_address=wifi_ip,
                    interface_type="wifi",
                ),
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address=eth_ip,
                    interface_type="ethernet",
                ),
            ]
        )

    # Add connections for both ethernet and wifi IPs
    for i, src in enumerate(node_ids):
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            eth_ip = _ethernet_ip(j)
            wifi_ip = f"192.168.1.{(j % 254) + 1}"
            # Add ethernet connection
            topology.add_connection(
                Connection(
                    source=src,
                    sink=dst,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(address=f"/ip4/{eth_ip}/tcp/8000")
                    ),
                )
            )
            # Add wifi connection
            topology.add_connection(
                Connection(
                    source=src,
                    sink=dst,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(address=f"/ip4/{wifi_ip}/tcp/8001")
                    ),
                )
            )

    cycle = Cycle(node_ids=node_ids)
    return cycle, topology, node_network, ephemeral_port, node_ids


# --- Property 6: Host resolution completeness with ethernet prioritization ---


class TestProperty6HostResolutionCompleteness:
    """Property 6: Host resolution completeness with ethernet prioritization.

    **Validates: Requirements 6.1, 6.3, 6.4**
    """

    @settings(max_examples=100)
    @given(data=topology_with_ethernet())
    def test_hosts_by_node_has_entry_for_every_node(self, data):
        """Verify: hosts_by_node has entry for every node in cycle."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        # Every node in the cycle must have an entry
        for nid in node_ids:
            assert nid in hosts_by_node, f"Node {nid} missing from hosts_by_node"

        # Each node's host list must have exactly world_size entries
        world_size = len(node_ids)
        for nid in node_ids:
            assert len(hosts_by_node[nid]) == world_size, (
                f"Node {nid} has {len(hosts_by_node[nid])} hosts, expected {world_size}"
            )

    @settings(max_examples=100)
    @given(data=topology_with_ethernet())
    def test_self_entry_is_bind_address(self, data):
        """Verify: each node's self-entry is Host(ip='0.0.0.0', port=ephemeral_port)."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        for rank, nid in enumerate(node_ids):
            self_host = hosts_by_node[nid][rank]
            assert self_host.ip == "0.0.0.0"
            assert self_host.port == ephemeral_port

    @settings(max_examples=100)
    @given(data=topology_with_ethernet())
    def test_all_non_self_entries_have_real_ips(self, data):
        """Verify: non-self entries have actual IPs (not placeholders)."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        for rank, nid in enumerate(node_ids):
            for idx, host in enumerate(hosts_by_node[nid]):
                if idx == rank:
                    continue
                # Must not be a placeholder
                assert host.ip != "198.51.100.1", (
                    f"Node {nid} has placeholder IP for node at index {idx}"
                )
                assert host.ip != "0.0.0.0", (
                    f"Non-self entry should not be 0.0.0.0"
                )
                assert host.port == ephemeral_port

    @settings(max_examples=100)
    @given(data=topology_with_mixed_interfaces())
    def test_ethernet_prioritized_over_wifi(self, data):
        """Verify: selected IPs are ethernet when available."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        for rank, nid in enumerate(node_ids):
            for idx, host in enumerate(hosts_by_node[nid]):
                if idx == rank:
                    continue
                # The selected IP should be the ethernet IP (10.1.1.x),
                # not the wifi IP (192.168.1.x)
                assert host.ip.startswith("10.1.1."), (
                    f"Node {nid} selected non-ethernet IP {host.ip} for node at index {idx}"
                )


# --- Property 7: MASTER_ADDR derivation from rank 0 ---


class TestProperty7MasterAddrDerivation:
    """Property 7: MASTER_ADDR derivation from rank 0.

    **Validates: Requirements 6.2**
    """

    @settings(max_examples=100)
    @given(data=topology_with_ethernet())
    def test_master_addr_is_rank_0_ethernet_ip(self, data):
        """Verify: derived MASTER_ADDR is always rank 0's ethernet IP."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        # Rank 0 is the first node in the cycle
        rank_0_node = node_ids[0]
        rank_0_ethernet_ip = _ethernet_ip(0)

        # For every non-rank-0 node, the IP at index 0 (rank 0's position)
        # should be rank 0's ethernet IP
        for rank, nid in enumerate(node_ids):
            if rank == 0:
                # Rank 0's own entry at index 0 is the bind address
                assert hosts_by_node[nid][0].ip == "0.0.0.0"
            else:
                # Other nodes should see rank 0's ethernet IP at index 0
                master_addr = hosts_by_node[nid][0].ip
                assert master_addr == rank_0_ethernet_ip, (
                    f"Node {nid} (rank {rank}) sees MASTER_ADDR={master_addr}, "
                    f"expected {rank_0_ethernet_ip}"
                )

    @settings(max_examples=100)
    @given(data=topology_with_ethernet())
    def test_master_port_equals_ephemeral_port(self, data):
        """Verify: derived MASTER_PORT always equals instance's ephemeral_port."""
        cycle, topology, node_network, ephemeral_port, node_ids = data

        hosts_by_node = get_pytorch_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=ephemeral_port,
            node_network=node_network,
        )

        # For every node, the port at rank 0's position should be ephemeral_port
        for rank, nid in enumerate(node_ids):
            master_port = hosts_by_node[nid][0].port
            assert master_port == ephemeral_port, (
                f"Node {nid} sees MASTER_PORT={master_port}, expected {ephemeral_port}"
            )
