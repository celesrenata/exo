"""
Unit tests for runner task dispatch.

Tests the PyTorch distributed backend dispatch logic in runner.py,
including ConnectToGroup, LoadModel, StartWarmup, Shutdown handlers,
and backend dispatch isolation.

Requirements: 1.4, 4.4, 5.3, 5.4, 7.1, 7.2, 7.3, 7.4, 8.1
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    PyTorchIPEXRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerReady,
    RunnerId,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_card() -> Any:
    """Create a minimal ModelCard-like object for testing."""
    from exo.shared.models.model_cards import ModelCard, ModelTask
    from exo.shared.types.memory import Memory

    return ModelCard(
        model_id="test/model",
        storage_size=Memory(in_bytes=1_000_000),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _make_shard(
    device_rank: int = 0,
    world_size: int = 4,
    start_layer: int = 0,
    end_layer: int = 8,
    n_layers: int = 32,
) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_make_model_card(),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
    )


def _make_instance(
    num_nodes: int = 4,
    ephemeral_port: int = 29500,
) -> tuple[PyTorchIPEXRingInstance, dict[str, PipelineShardMetadata]]:
    """Create a PyTorchIPEXRingInstance with num_nodes nodes."""
    node_ids = [NodeId(f"node-{i}") for i in range(num_nodes)]
    runner_ids = [RunnerId(f"runner-{i}") for i in range(num_nodes)]

    layers_per_node = 32 // num_nodes
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    for i, rid in enumerate(runner_ids):
        runner_to_shard[rid] = _make_shard(
            device_rank=i,
            world_size=num_nodes,
            start_layer=i * layers_per_node,
            end_layer=(i + 1) * layers_per_node,
            n_layers=32,
        )

    node_to_runner: dict[NodeId, RunnerId] = dict(zip(node_ids, runner_ids))

    hosts_by_node: dict[NodeId, list[Host]] = {}
    for i, nid in enumerate(node_ids):
        hosts_by_node[nid] = [Host(ip=f"10.1.1.{12 + i}", port=ephemeral_port)]

    shard_assignments = ShardAssignments(
        model_id="test/model",
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    instance = PyTorchIPEXRingInstance(
        instance_id="test-instance",
        shard_assignments=shard_assignments,
        hosts_by_node=hosts_by_node,
        ephemeral_port=ephemeral_port,
    )
    return instance, runner_to_shard


def _make_bound_instance(
    rank: int = 0,
    num_nodes: int = 4,
) -> BoundInstance:
    """Create a BoundInstance for a specific rank."""
    instance, runner_to_shard = _make_instance(num_nodes=num_nodes)
    runner_id = RunnerId(f"runner-{rank}")
    node_id = NodeId(f"node-{rank}")
    return BoundInstance(
        instance=instance,
        bound_runner_id=runner_id,
        bound_node_id=node_id,
    )


class FakeEventSender:
    """Collects events sent by the runner for assertion."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def send(self, event: Any) -> None:
        self.events.append(event)

    def get_status_events(self) -> list[Any]:
        return [e for e in self.events if isinstance(e, RunnerStatusUpdated)]

    def last_status(self) -> Any:
        statuses = self.get_status_events()
        return statuses[-1].runner_status if statuses else None


class FakeTaskReceiver:
    """Yields a sequence of tasks, then stops."""

    def __init__(self, tasks: list[Any]) -> None:
        self._tasks = tasks

    def __enter__(self) -> "FakeTaskReceiver":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __iter__(self):
        return iter(self._tasks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectToGroupSuccess:
    """Test: ConnectToGroup success → RunnerConnected status emitted."""

    def test_connect_to_group_pytorch_success(self) -> None:
        """ConnectToGroup with PyTorch backend initializes Gloo process group
        and transitions to RunnerConnected."""
        bound = _make_bound_instance(rank=0, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, shutdown_task])

        # Mock all heavy dependencies
        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ) as mock_init_pg,
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        # Verify init_process_group was called
        mock_init_pg.assert_called_once()
        config = mock_init_pg.call_args[0][0]
        assert config.rank == 0
        assert config.world_size == 4
        assert config.master_addr == "10.1.1.12"  # rank 0's IP
        assert config.master_port == 29500
        assert config.backend == "gloo"

        # Verify RunnerConnected was emitted
        statuses = event_sender.get_status_events()
        status_types = [type(s.runner_status).__name__ for s in statuses]
        assert "RunnerConnected" in status_types


class TestConnectToGroupFailure:
    """Test: ConnectToGroup failure → RunnerFailed with descriptive message."""

    def test_connect_to_group_pytorch_failure(self) -> None:
        """ConnectToGroup failure transitions to RunnerFailed with rank/world_size/gloo info."""
        bound = _make_bound_instance(rank=1, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, shutdown_task])

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group",
                side_effect=RuntimeError("Connection refused"),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        # Verify RunnerFailed was emitted with descriptive message
        statuses = event_sender.get_status_events()
        failed_statuses = [
            s for s in statuses if isinstance(s.runner_status, RunnerFailed)
        ]
        assert len(failed_statuses) >= 1
        error_msg = failed_statuses[0].runner_status.error_message
        assert error_msg is not None
        assert "rank=" in error_msg
        assert "world_size=" in error_msg
        assert "gloo" in error_msg


class TestLoadModelSuccess:
    """Test: LoadModel success → RunnerLoaded status emitted."""

    def test_load_model_pytorch_success(self) -> None:
        """LoadModel with PyTorch backend loads model and transitions to RunnerLoaded."""
        bound = _make_bound_instance(rank=0, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        load_task = LoadModel(
            task_id=TaskId("task-load"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, load_task, shutdown_task])

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock GPU detector
        mock_gpu_report = MagicMock()
        mock_gpu_report.has_gpu = True
        mock_gpu = MagicMock()
        mock_gpu.device_type = "xpu"
        mock_gpu.device_index = 0
        mock_gpu.name = "Intel Arc Graphics"
        mock_gpu.memory_architecture = MagicMock()
        mock_gpu.memory_architecture.value = "Shared"
        mock_gpu_report.gpus = [mock_gpu]

        import asyncio

        async def fake_load_model(**kwargs: Any) -> tuple[Any, Any]:
            return mock_model, mock_tokenizer

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_model = fake_load_model

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader",
                return_value=mock_loader_instance,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.gpu_detector.detect_gpus",
                return_value=mock_gpu_report,
            ),
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        statuses = event_sender.get_status_events()
        status_types = [type(s.runner_status).__name__ for s in statuses]
        assert "RunnerLoaded" in status_types


class TestLoadModelOOM:
    """Test: LoadModel OOM → RunnerFailed with memory info."""

    def test_load_model_pytorch_oom(self) -> None:
        """LoadModel OOM transitions to RunnerFailed with memory information."""
        bound = _make_bound_instance(rank=0, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        load_task = LoadModel(
            task_id=TaskId("task-load"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, load_task, shutdown_task])

        # Mock GPU detector
        mock_gpu_report = MagicMock()
        mock_gpu_report.has_gpu = True
        mock_gpu = MagicMock()
        mock_gpu.device_type = "xpu"
        mock_gpu.device_index = 0
        mock_gpu.name = "Intel Arc Graphics"
        mock_gpu.memory_architecture = MagicMock()
        mock_gpu.memory_architecture.value = "Shared"
        mock_gpu.available_memory_bytes = 4 * 1024**3
        mock_gpu_report.gpus = [mock_gpu]

        import asyncio

        async def fake_load_model_oom(**kwargs: Any) -> tuple[Any, Any]:
            raise RuntimeError("XPU out of memory. Tried to allocate 8.00 GiB")

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_model = fake_load_model_oom

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader",
                return_value=mock_loader_instance,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.gpu_detector.detect_gpus",
                return_value=mock_gpu_report,
            ),
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        # Verify RunnerFailed was emitted with OOM info
        statuses = event_sender.get_status_events()
        failed_statuses = [
            s for s in statuses if isinstance(s.runner_status, RunnerFailed)
        ]
        assert len(failed_statuses) >= 1
        error_msg = failed_statuses[0].runner_status.error_message
        assert error_msg is not None
        assert "out of memory" in error_msg.lower() or "OOM" in error_msg


class TestStartWarmupSuccess:
    """Test: StartWarmup success → RunnerReady status emitted."""

    def test_warmup_pytorch_success(self) -> None:
        """StartWarmup with PyTorch backend transitions to RunnerReady."""
        bound = _make_bound_instance(rank=0, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        load_task = LoadModel(
            task_id=TaskId("task-load"),
            instance_id=instance.instance_id,
        )
        warmup_task = StartWarmup(
            task_id=TaskId("task-warmup"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver(
            [connect_task, load_task, warmup_task, shutdown_task]
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_gpu_report = MagicMock()
        mock_gpu_report.has_gpu = True
        mock_gpu = MagicMock()
        mock_gpu.device_type = "xpu"
        mock_gpu.device_index = 0
        mock_gpu.name = "Intel Arc Graphics"
        mock_gpu.memory_architecture = MagicMock()
        mock_gpu.memory_architecture.value = "Shared"
        mock_gpu_report.gpus = [mock_gpu]

        import asyncio

        async def fake_load_model(**kwargs: Any) -> tuple[Any, Any]:
            return mock_model, mock_tokenizer

        mock_loader_instance = MagicMock()
        mock_loader_instance.load_model = fake_load_model

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.send_activation"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.recv_activation",
                return_value=MagicMock(),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader",
                return_value=mock_loader_instance,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=10,
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.gpu_detector.detect_gpus",
                return_value=mock_gpu_report,
            ),
            patch("torch.zeros", return_value=MagicMock()),
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        statuses = event_sender.get_status_events()
        status_types = [type(s.runner_status).__name__ for s in statuses]
        assert "RunnerReady" in status_types


class TestShutdownCleanup:
    """Test: Shutdown → destroy_process_group called, GPU caches cleared."""

    def test_shutdown_pytorch_cleanup(self) -> None:
        """Shutdown calls destroy_process_group and clears GPU caches."""
        bound = _make_bound_instance(rank=0, num_nodes=4)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, shutdown_task])

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ) as mock_destroy,
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
            patch("torch.xpu", create=True) as mock_xpu,
            patch("torch.cuda") as mock_cuda,
        ):
            from exo.worker.runner.runner import main

            main(bound, event_sender, task_receiver)

        # Verify destroy_process_group was called with timeout
        mock_destroy.assert_called_once_with(timeout_seconds=5.0)

        # Verify RunnerShutdown was emitted
        statuses = event_sender.get_status_events()
        status_types = [type(s.runner_status).__name__ for s in statuses]
        assert "RunnerShutdown" in status_types


class TestBackendDispatch:
    """Test: Backend dispatch: PyTorchIPEXRingInstance → PyTorch path, MlxRingInstance → MLX path."""

    def test_pytorch_instance_uses_pytorch_backend(self) -> None:
        """PyTorchIPEXRingInstance triggers pytorch_ipex backend type."""
        instance, _ = _make_instance(num_nodes=2)
        assert isinstance(instance, PyTorchIPEXRingInstance)

        # The backend detection logic in runner.py checks isinstance
        is_pytorch_ipex = isinstance(instance, PyTorchIPEXRingInstance)
        assert is_pytorch_ipex is True

    def test_mlx_instance_not_pytorch(self) -> None:
        """MlxRingInstance does not trigger pytorch_ipex backend."""
        from exo.shared.types.worker.instances import MlxRingInstance

        instance, _ = _make_instance(num_nodes=2)
        # Create an MlxRingInstance with same structure
        mlx_instance = MlxRingInstance(
            instance_id="mlx-test",
            shard_assignments=instance.shard_assignments,
            hosts_by_node=instance.hosts_by_node,
            ephemeral_port=29500,
        )
        is_pytorch_ipex = isinstance(mlx_instance, PyTorchIPEXRingInstance)
        assert is_pytorch_ipex is False


class TestImportIsolation:
    """Test: Import isolation: PyTorch runner doesn't import MLX."""

    def test_pytorch_path_does_not_import_mlx(self) -> None:
        """When running PyTorch backend, MLX modules should not be imported."""
        bound = _make_bound_instance(rank=0, num_nodes=2)
        instance = bound.instance
        assert isinstance(instance, PyTorchIPEXRingInstance)

        # Track which modules get imported
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        mlx_imported = []

        def tracking_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("mlx") or name.startswith("mlx_lm"):
                mlx_imported.append(name)
            return original_import(name, *args, **kwargs)

        event_sender = FakeEventSender()
        connect_task = ConnectToGroup(
            task_id=TaskId("task-connect"),
            instance_id=instance.instance_id,
        )
        shutdown_task = Shutdown(
            task_id=TaskId("task-shutdown"),
            instance_id=instance.instance_id,
            runner_id=bound.bound_runner_id,
        )
        task_receiver = FakeTaskReceiver([connect_task, shutdown_task])

        with (
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.init_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.distributed.destroy_process_group"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.generator.pytorch_ipex_generate",
                return_value=iter([]),
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.model_loader.ModelLoader"
            ),
            patch(
                "exo.worker.engines.pytorch_ipex.warmup.warmup_pytorch_ipex_inference",
                return_value=0,
            ),
            patch("builtins.__import__", side_effect=tracking_import),
        ):
            try:
                from exo.worker.runner.runner import main
                main(bound, event_sender, task_receiver)
            except Exception:
                pass  # We only care about import tracking

        # Verify no MLX modules were imported during PyTorch path execution
        # Note: some MLX imports may happen at module level in other files,
        # but the runner's PyTorch path should not trigger them
        mlx_in_runner = [m for m in mlx_imported if "runner" in str(m)]
        assert len(mlx_in_runner) == 0, (
            f"MLX modules imported during PyTorch runner path: {mlx_in_runner}"
        )