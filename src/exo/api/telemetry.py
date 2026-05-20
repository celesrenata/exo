"""Telemetry streaming API routes.

Provides endpoints for real-time cluster telemetry:
- GET /api/telemetry/cluster: Returns latest snapshot of all node metrics as JSON.
- GET /api/telemetry/stream: SSE stream of per-node telemetry updates.
"""

import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    WouldBlock,
    create_memory_object_stream,
)
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastapi import Request
from fastapi.responses import StreamingResponse

from exo.api.keepalive import with_sse_keepalive
from exo.shared.types.common import NodeId
from exo.telemetry.models import ClusterTelemetry, NodeTelemetry

if TYPE_CHECKING:
    from exo.api.main import API

_STALENESS_THRESHOLD = timedelta(seconds=3)


class TelemetryAggregator:
    """Collects per-node telemetry reports and broadcasts updates to SSE subscribers.

    Maintains the latest snapshot per node with timestamps. Nodes are marked stale
    after 3 seconds without an update. Nodes that have never reported are omitted
    entirely from responses.
    """

    def __init__(self) -> None:
        self._latest: dict[NodeId, NodeTelemetry] = {}
        self._timestamps: dict[NodeId, datetime] = {}
        self._subscribers: list[MemoryObjectSendStream[str]] = []

    async def report(self, telemetry: NodeTelemetry) -> None:
        """Store a new telemetry report and notify all SSE subscribers."""
        self._latest[telemetry.node_id] = telemetry
        self._timestamps[telemetry.node_id] = datetime.now(timezone.utc)

        event_payload = json.dumps(
            {
                "node_id": str(telemetry.node_id),
                "gpu": telemetry.gpu.model_dump(mode="json"),
                "network": telemetry.network.model_dump(mode="json"),
                "is_stale": False,
            }
        )
        await self._broadcast(event_payload)

    def get_cluster_snapshot(self) -> ClusterTelemetry:
        """Return the current cluster telemetry snapshot with staleness markers."""
        now = datetime.now(timezone.utc)
        nodes: dict[NodeId, NodeTelemetry] = {}
        stale_nodes: list[NodeId] = []

        for node_id, telemetry in self._latest.items():
            timestamp = self._timestamps[node_id]
            if now - timestamp > _STALENESS_THRESHOLD:
                stale_nodes.append(node_id)
            else:
                nodes[node_id] = telemetry

        return ClusterTelemetry(nodes=nodes, stale_nodes=stale_nodes)

    def subscribe(self) -> tuple[MemoryObjectSendStream[str], MemoryObjectReceiveStream[str]]:
        """Create a new subscriber stream pair. Returns (send, receive)."""
        send, receive = create_memory_object_stream[str](max_buffer_size=64)
        self._subscribers.append(send)
        return send, receive

    async def _broadcast(self, payload: str) -> None:
        """Send the payload to all active subscribers, removing disconnected ones."""
        active: list[MemoryObjectSendStream[str]] = []
        for send in self._subscribers:
            try:
                send.send_nowait(payload)
                active.append(send)
            except (BrokenResourceError, ClosedResourceError, WouldBlock):
                # Subscriber disconnected or buffer full — drop it
                pass
        self._subscribers = active


def register_telemetry_routes(api: "API") -> None:
    """Register GET /api/telemetry/cluster and GET /api/telemetry/stream on the API app."""
    aggregator = TelemetryAggregator()
    api.telemetry_aggregator = aggregator  # pyright: ignore[reportAttributeAccessIssue]

    @api.app.get("/api/telemetry/cluster")
    async def get_cluster_telemetry() -> ClusterTelemetry:  # pyright: ignore[reportUnusedFunction]
        return aggregator.get_cluster_snapshot()

    @api.app.post("/api/telemetry/report")
    async def report_telemetry(request: Request) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        body = await request.body()
        telemetry = NodeTelemetry.model_validate_json(body)
        await aggregator.report(telemetry)
        return {"status": "ok"}

    @api.app.get("/api/telemetry/stream")
    async def stream_telemetry(request: Request) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        _send, receive = aggregator.subscribe()

        async def event_generator() -> AsyncIterator[str]:
            try:
                async with receive:
                    async for payload in receive:
                        if await request.is_disconnected():
                            break
                        yield f"event: telemetry\ndata: {payload}\n\n"
            finally:
                # Remove the send channel from subscribers on disconnect
                if _send in aggregator._subscribers:  # pyright: ignore[reportPrivateUsage]
                    aggregator._subscribers.remove(_send)  # pyright: ignore[reportPrivateUsage]

        return StreamingResponse(
            with_sse_keepalive(event_generator()),
            media_type="text/event-stream",
        )
