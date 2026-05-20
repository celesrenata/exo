"""Pydantic models for telemetry data structures."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel


class GpuMetrics(FrozenModel):
    """GPU telemetry for a single node."""

    node_id: NodeId
    timestamp: datetime
    frequency_mhz: int | None = None
    utilization_percent: float | None = None
    render_busy_percent: float | None = None
    memory_bandwidth_percent: float | None = None
    source: Literal["intel_gpu_top", "sysfs", "unavailable"] = "intel_gpu_top"


class NetworkMetrics(FrozenModel):
    """Network telemetry for a single node's cluster interface."""

    node_id: NodeId
    timestamp: datetime
    interface_name: str
    bytes_sent: int
    bytes_received: int
    throughput_sent_bytes_per_sec: float
    throughput_received_bytes_per_sec: float
    latency_ms: float | None = None


class NodeTelemetry(FrozenModel):
    """Combined telemetry report from a single node."""

    node_id: NodeId
    gpu: GpuMetrics
    network: NetworkMetrics


class ClusterTelemetry(FrozenModel):
    """Snapshot of all node telemetry."""

    nodes: Mapping[NodeId, NodeTelemetry]
    stale_nodes: Sequence[NodeId]
