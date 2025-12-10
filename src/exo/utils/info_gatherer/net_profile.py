import socket
from collections.abc import Mapping
from ipaddress import ip_address

from anyio import create_task_group, to_thread

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodePerformanceProfile


# TODO: ref. api port
async def check_reachability(
    target_ip: str, target_node_id: NodeId, out: dict[NodeId, set[str]]
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # 1 second timeout
    try:
        result = await to_thread.run_sync(sock.connect_ex, (target_ip, 52415))
    except socket.gaierror:
        # seems to throw on ipv6 loopback. oh well
        # logger.warning(f"invalid {target_ip=}")
        return
    finally:
        sock.close()

    if result == 0:
        if target_node_id not in out:
            out[target_node_id] = set()
        out[target_node_id].add(target_ip)


async def check_reachable(
    our_node_id: NodeId,
    topology: Topology,
    profiles: Mapping[NodeId, NodePerformanceProfile],
) -> Mapping[NodeId, set[str]]:
    reachable: dict[NodeId, set[str]] = {}
    our_profile = profiles.get(our_node_id, None)
    if our_profile is None:
        return {}
    our_interfaces = our_profile.network_interfaces
    async with create_task_group() as tg:
        for node_id in topology.list_nodes():
            if node_id not in profiles or node_id == our_node_id:
                continue
            for iface in profiles[node_id].network_interfaces:
                if ip_address(iface.ip_address).is_loopback:
                    # Definitely a loopback address
                    continue
                if iface in our_interfaces:
                    # Skip duplicates with our own interfaces
                    continue
                tg.start_soon(check_reachability, iface.ip_address, node_id, reachable)

    return reachable
