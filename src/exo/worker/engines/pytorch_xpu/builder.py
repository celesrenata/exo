"""PyTorch XPU Builder — implements the Builder protocol for the upstream runner.

Drives the ConnectToGroup → LoadModel → build() lifecycle, producing a
PyTorchXPUEngine that the runner can use for generation.

Gloo backend only. All broadcast tensors must be on CPU.
Device is xpu:0 (Intel Arc iGPU via torch.xpu).
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance, PyTorchXPURingInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger


def _resolve_master_addr(bound_instance: BoundInstance) -> str:
    """Determine master_addr for Gloo TCPStore rendezvous.

    Uses the MASTER_ADDR env var (set to 10.1.1.12 by the NixOS service)
    as a fixed rendezvous point. The node at that IP hosts the TCPStore
    regardless of its model rank.

    If MASTER_ADDR is not set, falls back to detecting own bond0 IP
    (works when rank 0 is on the current node).
    """
    import os
    import socket

    # Use MASTER_ADDR env var as fixed rendezvous
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        return master_addr

    # Fallback: detect own bond0 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.1.1.1", 1))
        our_ip = s.getsockname()[0]
        s.close()
        return our_ip
    except Exception:
        return "10.1.1.12"


@dataclass
class PyTorchXPUBuilder(Builder):
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]

    # Populated during connect/load
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _rank: int = field(default=0, init=False)
    _world_size: int = field(default=1, init=False)
    _device: str = field(default="xpu:0", init=False)

    def connect(self, bound_instance: BoundInstance) -> None:
        """Initialize Gloo process group using the instance's hosts and shard metadata."""
        import torch.distributed as dist

        shard = bound_instance.bound_shard
        self._rank = shard.device_rank
        self._world_size = shard.world_size

        if self._world_size <= 1:
            logger.info("Single-node XPU instance, skipping process group init")
            return

        assert isinstance(bound_instance.instance, (PyTorchXPURingInstance, MlxRingInstance))
        master_addr = _resolve_master_addr(bound_instance)
        master_port = bound_instance.instance.ephemeral_port

        logger.info(
            f"PyTorchXPUBuilder.connect: rank={self._rank}, world_size={self._world_size}, "
            f"master_addr={master_addr}, master_port={master_port}"
        )

        from exo.worker.engines.pytorch_xpu.distributed import (
            ProcessGroupConfig,
            init_process_group,
        )

        config = ProcessGroupConfig(
            rank=self._rank,
            world_size=self._world_size,
            master_addr=master_addr,
            master_port=master_port,
            init_timeout_seconds=600,  # 10 min — needed for Qwen3.5/3.6 prefill
        )
        init_process_group(config)

        # Also set the TP group handle so TensorParallelShard can use it
        from exo.worker.engines.pytorch_xpu.distributed import (
            _tp_process_group,
        )

        # If init_process_group set up the default group, point TP group at WORLD
        import exo.worker.engines.pytorch_xpu.distributed as dist_module

        if dist_module._tp_process_group is None:
            dist_module._tp_process_group = dist.group.WORLD

        logger.info("PyTorchXPUBuilder.connect: process group initialized")

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        """Load the HuggingFace model onto XPU device, yielding progress."""
        import torch

        shard = bound_instance.bound_shard
        model_id = str(shard.model_card.model_id)

        # Determine device — use XPU if available, else CPU
        if torch.xpu.is_available():
            self._device = "xpu:0"
        else:
            logger.warning("torch.xpu not available, falling back to CPU")
            self._device = "cpu"

        logger.info(
            f"PyTorchXPUBuilder.load: model={model_id}, device={self._device}, "
            f"layers=[{shard.start_layer}, {shard.end_layer}), "
            f"rank={self._rank}/{self._world_size}"
        )

        from exo.worker.engines.pytorch_xpu.model_loader import ModelLoader

        loader = ModelLoader()

        # ModelLoader.load_model is async, but we're in a sync generator context.
        # Use the synchronous _load_model_sync method directly.
        device = torch.device(self._device)

        # Yield initial progress
        total_layers = shard.end_layer - shard.start_layer
        yield ModelLoadingResponse(layers_loaded=0, total=total_layers)

        # Load model synchronously
        model, tokenizer = loader._load_model_sync(
            model_id=model_id,
            device=device,
            shard_metadata=shard,
        )

        self._model = model
        self._tokenizer = tokenizer

        # Yield completion
        yield ModelLoadingResponse(layers_loaded=total_layers, total=total_layers)

        logger.info(f"PyTorchXPUBuilder.load: model loaded successfully on {self._device}")

    def build(self) -> Engine:
        """Return a PyTorchXPUEngine instance."""
        assert self._model is not None, "Must call load() before build()"
        assert self._tokenizer is not None, "Must call load() before build()"

        from exo.worker.engines.pytorch_xpu.engine import PyTorchXPUEngine

        return PyTorchXPUEngine(
            model=self._model,
            tokenizer=self._tokenizer,
            rank=self._rank,
            world_size=self._world_size,
            device=self._device,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
        )

    def close(self) -> None:
        """Clean up model and process group."""
        with contextlib.suppress(Exception):
            del self._model
            self._model = None
        with contextlib.suppress(Exception):
            del self._tokenizer
            self._tokenizer = None

        try:
            import torch.distributed as dist

            if dist.is_initialized():
                from exo.worker.engines.pytorch_xpu.distributed import (
                    destroy_process_group,
                )

                destroy_process_group()
        except Exception:
            pass

        logger.info("PyTorchXPUBuilder.close: cleanup complete")
