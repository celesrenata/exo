"""Telemetry aggregation module for the exo cluster.

This module provides functions to aggregate telemetry data from
multiple nodes into a single cluster telemetry report.
"""

from collections.abc import Mapping, Sequence

from exo.shared.types.common import NodeId
from exo.telemetry.models import ClusterTelemetry, NodeTelemetry


def build_cluster_telemetry(
    reporting_nodes: Mapping[NodeId, NodeTelemetry],
    all_known_node_ids: Sequence[NodeId],
) -> ClusterTelemetry:
    """Aggregate telemetry from reporting nodes into a cluster-wide view.

    Only nodes present in ``reporting_nodes`` are included in the response.
    Nodes in ``all_known_node_ids`` that did not report metrics (offline
    nodes) are omitted entirely rather than returning zero values.

    Args:
        reporting_nodes: A mapping of node IDs to their telemetry data for
            nodes that have reported.
        all_known_node_ids: A sequence of all node IDs that are part of the
            cluster, regardless of whether they have reported.

    Returns:
        A ClusterTelemetry instance containing the telemetry of all
        reporting nodes and an empty list of stale nodes.
    """
    return ClusterTelemetry(nodes=dict(reporting_nodes), stale_nodes=[])
