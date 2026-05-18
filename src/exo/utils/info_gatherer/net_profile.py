from collections import defaultdict
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
    api_port: int,
) -> None:
    """Check if a node is reachable at the given IP and verify its identity."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{api_port}/node_id"
    else:
        url = f"http://{target_ip}:{api_port}/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None:
        return

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(target_ip)


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    api_port: int,
) -> AsyncGenerator[tuple[str, NodeId], None]:
    """Yield (ip, node_id) pairs as reachability probes complete."""

    send, recv = channel[tuple[str, NodeId]]()

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[tuple[str, NodeId]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            try:
                await check_reachability(target_ip, expected_node_id, out, client, api_port)
            except Exception as e:
                logger.warning(f"Probe to {target_ip} for {expected_node_id} failed: {e}")
                return
            if expected_node_id in out:
                logger.debug(f"Probe succeeded: {target_ip} -> {expected_node_id}")
                await send.send((target_ip, expected_node_id))

    def _is_usable_remote_ip(ip: str) -> bool:
        """Filter out loopback, link-local, and container/overlay network addresses."""
        if ip.startswith("127."):
            return False
        if ip == "::1":
            return False
        if ip.startswith("fe80:"):
            return False
        if ip.startswith("169.254."):
            return False
        # Filter container/overlay networks (Docker, Flannel, CNI, etc.)
        if ip.startswith("172.17."):  # Docker default bridge
            return False
        if ip.startswith("10.42."):  # Flannel/K8s pod network
            return False
        if ip.startswith("10.43."):  # K8s service network
            return False
        if ip.startswith("10.244."):  # K8s pod CIDR
            return False
        # Filter IPv6 that isn't global unicast
        if ":" in ip and not ip.startswith("2") and not ip.startswith("3"):
            return False
        return True

    probed_count = 0
    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                if _is_usable_remote_ip(iface.ip_address):
                    probed_count += 1
                    tg.start_soon(_probe, iface.ip_address, node_id, client, send.clone())
        send.close()
        logger.debug(f"Net profiler: probing {probed_count} IPs across {len(list(topology.list_nodes())) - 1} remote nodes")

        with recv:
            async for item in recv:
                yield item
