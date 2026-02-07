from datetime import datetime
from typing import final

from pydantic import Field

from exo.shared.topology import Connection
from exo.shared.types.chunks import GenerationChunk, InputImageChunk
from exo.shared.types.common import CommandId, Id, NodeId, SessionId
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.pydantic_ext import CamelCaseModel, FrozenModel, TaggedModel


class EventId(Id):
    """
    Newtype around `ID`
    """


class BaseEvent(TaggedModel):
    event_id: EventId = Field(default_factory=EventId)
    # Internal, for debugging. Please don't rely on this field for anything!
    _master_time_stamp: None | datetime = None


class TestEvent(BaseEvent):
    __test__ = False


class TaskCreated(BaseEvent):
    task_id: TaskId
    task: Task


class TaskAcknowledged(BaseEvent):
    task_id: TaskId


class TaskDeleted(BaseEvent):
    task_id: TaskId


class TaskStatusUpdated(BaseEvent):
    task_id: TaskId
    task_status: TaskStatus


class TaskFailed(BaseEvent):
    task_id: TaskId
    error_type: str
    error_message: str


class InstanceCreated(BaseEvent):
    instance: Instance

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InstanceCreated):
            return self.instance == other.instance and self.event_id == other.event_id

        return False


class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


class RunnerStatusUpdated(BaseEvent):
    runner_id: RunnerId
    runner_status: RunnerStatus


class RunnerDeleted(BaseEvent):
    runner_id: RunnerId


class BackendInitialized(BaseEvent):
    """Event emitted when a backend successfully initializes.

    This event provides observability into which backend and device
    are being used for inference on each runner.
    """

    runner_id: RunnerId
    backend_type: str  # "mlx", "tinygrad", "npu"
    device_type: str  # "CPU", "GPU", "NPU", "METAL"
    device_name: str
    runtime: str | None  # "LEVEL_ZERO", "OPENCL", "METAL", "CUDA", None


class BackendFailed(BaseEvent):
    """Event emitted when a backend fails to initialize.

    This event is used for debugging backend initialization failures
    and tracking fallback behavior.
    """

    runner_id: RunnerId
    backend_type: str  # "mlx", "tinygrad", "npu"
    error_message: str
    fallback_backend: str | None  # Backend that will be tried next, if any


class GPUMetricsCollected(BaseEvent):
    """Event emitted when GPU performance metrics are collected.

    This event provides observability into GPU utilization and memory
    usage during inference for dashboard visibility.
    """

    runner_id: RunnerId
    device_name: str
    runtime: str  # "LEVEL_ZERO", "OPENCL", "CUDA", "METAL"
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float | None  # None if unavailable
    timestamp: float  # Unix timestamp


class NodeTimedOut(BaseEvent):
    node_id: NodeId


# TODO: bikeshed this name
class NodeGatheredInfo(BaseEvent):
    node_id: NodeId
    when: str  # this is a manually cast datetime overrode by the master when the event is indexed, rather than the local time on the device
    info: GatheredInfo


class NodeDownloadProgress(BaseEvent):
    download_progress: DownloadProgress


class ChunkGenerated(BaseEvent):
    command_id: CommandId
    chunk: GenerationChunk


class InputChunkReceived(BaseEvent):
    command_id: CommandId
    chunk: InputImageChunk


class TopologyEdgeCreated(BaseEvent):
    conn: Connection


class TopologyEdgeDeleted(BaseEvent):
    conn: Connection


@final
class TraceEventData(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


@final
class TracesCollected(BaseEvent):
    task_id: TaskId
    rank: int
    traces: list[TraceEventData]


@final
class TracesMerged(BaseEvent):
    task_id: TaskId
    traces: list[TraceEventData]


Event = (
    TestEvent
    | TaskCreated
    | TaskStatusUpdated
    | TaskFailed
    | TaskDeleted
    | TaskAcknowledged
    | InstanceCreated
    | InstanceDeleted
    | RunnerStatusUpdated
    | RunnerDeleted
    | BackendInitialized
    | BackendFailed
    | GPUMetricsCollected
    | NodeTimedOut
    | NodeGatheredInfo
    | NodeDownloadProgress
    | ChunkGenerated
    | InputChunkReceived
    | TopologyEdgeCreated
    | TopologyEdgeDeleted
    | TracesCollected
    | TracesMerged
)


class IndexedEvent(CamelCaseModel):
    """An event indexed by the master, with a globally unique index"""

    idx: int = Field(ge=0)
    event: Event


class ForwarderEvent(CamelCaseModel):
    """An event the forwarder will serialize and send over the network"""

    origin_idx: int = Field(ge=0)
    origin: NodeId
    session: SessionId
    event: Event
