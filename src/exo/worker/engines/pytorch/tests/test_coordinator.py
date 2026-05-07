"""Unit tests for pipeline/coordinator.py — pipeline orchestration.

Tests the PipelineCoordinator class using mock engine and communicator
implementations to verify pipeline forwarding logic, token generation
coordination, and error handling for unreachable nodes.

Requirements: 5.4, 5.7, 5.8, 5.9
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from exo.worker.engines.pytorch.pipeline.coordinator import (
    PipelineConfig,
    PipelineCoordinator,
    PipelineNodeUnreachableError,
    _describe_position,
)


# ---------------------------------------------------------------------------
# Mock engine and communicator for testing
# ---------------------------------------------------------------------------


class MockEngine:
    """Mock engine that tracks calls and returns predictable tensors."""

    def __init__(self, hidden_size: int = 64, dtype: torch.dtype = torch.float32) -> None:
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._device = "cpu"
        self.calls: list[str] = []

    def forward_layers(
        self, hidden_states: torch.Tensor, start_layer: int, end_layer: int
    ) -> torch.Tensor:
        self.calls.append(f"forward_layers({start_layer}, {end_layer})")
        # Return same shape, slightly modified to verify processing
        return hidden_states + 1.0

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append("embed_tokens")
        batch_size = token_ids.shape[0] if token_ids.dim() > 0 else 1
        return torch.zeros(batch_size, self._hidden_size, dtype=self._dtype)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append("lm_head")
        # Return logits of vocab size 100
        batch_size = hidden_states.shape[0]
        return torch.randn(batch_size, 100, dtype=self._dtype)

    def sample_token(self, logits: torch.Tensor) -> np.ndarray:
        self.calls.append("sample_token")
        return np.array([42], dtype=np.int64)

    @property
    def device(self) -> str:
        return self._device

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


@dataclass
class SentTensor:
    """Record of a tensor sent via the mock communicator."""

    tensor: torch.Tensor
    dst_rank: int


class MockCommunicator:
    """Mock communicator that records sends and returns stored tensors on recv."""

    def __init__(self) -> None:
        self.sent: list[SentTensor] = []
        self._recv_tensors: list[torch.Tensor] = []
        self.raise_on_send: bool = False
        self.raise_on_recv: bool = False

    def queue_recv_tensor(self, tensor: torch.Tensor) -> None:
        """Queue a tensor to be returned on the next recv_tensor call."""
        self._recv_tensors.append(tensor)

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        if self.raise_on_send:
            raise TimeoutError(f"Timeout sending to rank {dst_rank}")
        self.sent.append(SentTensor(tensor=tensor, dst_rank=dst_rank))

    def recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        target_device: str,
    ) -> torch.Tensor:
        if self.raise_on_recv:
            raise TimeoutError(f"Timeout receiving from rank {src_rank}")
        if self._recv_tensors:
            return self._recv_tensors.pop(0)
        # Return a tensor of the requested shape
        return torch.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(rank: int, world_size: int = 4, total_layers: int = 28) -> PipelineConfig:
    return PipelineConfig(
        world_size=world_size,
        rank=rank,
        total_layers=total_layers,
        master_addr="10.1.1.12",
        master_port=29500,
        transport="ethernet",
    )


def _make_coordinator(
    rank: int, world_size: int = 4, total_layers: int = 28
) -> tuple[PipelineCoordinator, MockEngine, MockCommunicator]:
    config = _make_config(rank, world_size, total_layers)
    engine = MockEngine()
    communicator = MockCommunicator()
    coordinator = PipelineCoordinator(config, engine, communicator)
    return coordinator, engine, communicator


# ---------------------------------------------------------------------------
# Tests: PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for the PipelineConfig frozen dataclass."""

    def test_frozen_immutability(self) -> None:
        config = _make_config(0)
        with pytest.raises(Exception):  # FrozenInstanceError
            config.rank = 1  # type: ignore[misc]

    def test_fields_stored_correctly(self) -> None:
        config = PipelineConfig(
            world_size=3,
            rank=1,
            total_layers=32,
            master_addr="192.168.1.1",
            master_port=12345,
            transport="rdma",
        )
        assert config.world_size == 3
        assert config.rank == 1
        assert config.total_layers == 32
        assert config.master_addr == "192.168.1.1"
        assert config.master_port == 12345
        assert config.transport == "rdma"


# ---------------------------------------------------------------------------
# Tests: get_stage_assignment
# ---------------------------------------------------------------------------


class TestGetStageAssignment:
    """Tests for PipelineCoordinator.get_stage_assignment()."""

    def test_rank_0_of_4_nodes_28_layers(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=0, world_size=4, total_layers=28)
        start, end = coordinator.get_stage_assignment()
        assert start == 0
        assert end == 7

    def test_rank_3_of_4_nodes_28_layers(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=3, world_size=4, total_layers=28)
        start, end = coordinator.get_stage_assignment()
        assert start == 21
        assert end == 28

    def test_rank_1_of_2_nodes_10_layers(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=1, world_size=2, total_layers=10)
        start, end = coordinator.get_stage_assignment()
        assert start == 5
        assert end == 10

    def test_uneven_division_rank_0_gets_extra(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=0, world_size=4, total_layers=30)
        start, end = coordinator.get_stage_assignment()
        # 30 / 4 = 7 remainder 2, so ranks 0 and 1 get 8 layers
        assert start == 0
        assert end == 8

    def test_supports_world_size_2(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=0, world_size=2, total_layers=20)
        start, end = coordinator.get_stage_assignment()
        assert start == 0
        assert end == 10

    def test_supports_world_size_3(self) -> None:
        coordinator, _, _ = _make_coordinator(rank=2, world_size=3, total_layers=12)
        start, end = coordinator.get_stage_assignment()
        assert start == 8
        assert end == 12


# ---------------------------------------------------------------------------
# Tests: forward_pipeline
# ---------------------------------------------------------------------------


class TestForwardPipeline:
    """Tests for PipelineCoordinator.forward_pipeline()."""

    @pytest.mark.asyncio
    async def test_first_stage_embeds_and_sends(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=0, world_size=4)
        input_tensor = torch.tensor([1, 2, 3], dtype=torch.long)

        result = await coordinator.forward_pipeline(input_tensor, "req-1")

        # First stage returns None (not last stage)
        assert result is None
        # Engine should have embedded and forwarded
        assert "embed_tokens" in engine.calls
        assert "forward_layers(0, 7)" in engine.calls
        # Should have sent to rank 1
        assert len(communicator.sent) == 1
        assert communicator.sent[0].dst_rank == 1

    @pytest.mark.asyncio
    async def test_last_stage_receives_and_returns_logits(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=3, world_size=4)
        # Queue a tensor for the recv
        communicator.queue_recv_tensor(torch.zeros(1, 64))

        result = await coordinator.forward_pipeline(torch.empty(0), "req-1")

        # Last stage returns logits
        assert result is not None
        assert result.shape[1] == 100  # vocab size from mock
        # Engine should have forwarded and applied lm_head
        assert "forward_layers(21, 28)" in engine.calls
        assert "lm_head" in engine.calls
        # Should NOT have sent to anyone
        assert len(communicator.sent) == 0

    @pytest.mark.asyncio
    async def test_middle_stage_receives_and_sends(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=1, world_size=4)
        communicator.queue_recv_tensor(torch.zeros(1, 64))

        result = await coordinator.forward_pipeline(torch.empty(0), "req-1")

        # Middle stage returns None
        assert result is None
        # Should have forwarded through local layers
        assert "forward_layers(7, 14)" in engine.calls
        # Should have sent to rank 2
        assert len(communicator.sent) == 1
        assert communicator.sent[0].dst_rank == 2

    @pytest.mark.asyncio
    async def test_two_node_pipeline_first_stage(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=0, world_size=2, total_layers=10)
        input_tensor = torch.tensor([5], dtype=torch.long)

        result = await coordinator.forward_pipeline(input_tensor, "req-2")

        assert result is None
        assert "embed_tokens" in engine.calls
        assert "forward_layers(0, 5)" in engine.calls
        assert len(communicator.sent) == 1
        assert communicator.sent[0].dst_rank == 1

    @pytest.mark.asyncio
    async def test_two_node_pipeline_last_stage(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=1, world_size=2, total_layers=10)
        communicator.queue_recv_tensor(torch.zeros(1, 64))

        result = await coordinator.forward_pipeline(torch.empty(0), "req-2")

        assert result is not None
        assert "forward_layers(5, 10)" in engine.calls
        assert "lm_head" in engine.calls


# ---------------------------------------------------------------------------
# Tests: generate_token
# ---------------------------------------------------------------------------


class TestGenerateToken:
    """Tests for PipelineCoordinator.generate_token()."""

    @pytest.mark.asyncio
    async def test_rank_0_returns_empty_array(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=0, world_size=4)
        prompt = np.array([1, 2, 3], dtype=np.int64)

        result = await coordinator.generate_token(prompt)

        # Rank 0 is not the last stage, returns empty
        assert len(result) == 0
        assert "embed_tokens" in engine.calls

    @pytest.mark.asyncio
    async def test_last_rank_returns_sampled_token(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=3, world_size=4)
        communicator.queue_recv_tensor(torch.zeros(1, 64))
        prompt = np.array([1, 2, 3], dtype=np.int64)

        result = await coordinator.generate_token(prompt)

        # Last rank samples and returns token
        assert len(result) == 1
        assert result[0] == 42  # From mock
        assert "sample_token" in engine.calls

    @pytest.mark.asyncio
    async def test_middle_rank_returns_empty_array(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=2, world_size=4)
        communicator.queue_recv_tensor(torch.zeros(1, 64))
        prompt = np.array([1, 2, 3], dtype=np.int64)

        result = await coordinator.generate_token(prompt)

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Error handling — unreachable nodes (Requirement 5.9)
# ---------------------------------------------------------------------------


class TestUnreachableNodes:
    """Tests for failure reporting when nodes become unreachable."""

    @pytest.mark.asyncio
    async def test_send_failure_reports_node_identity(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=0, world_size=4)
        communicator.raise_on_send = True
        input_tensor = torch.tensor([1, 2, 3], dtype=torch.long)

        with pytest.raises(PipelineNodeUnreachableError) as exc_info:
            await coordinator.forward_pipeline(input_tensor, "req-fail")

        error = exc_info.value
        assert error.rank == 1  # Next rank
        assert error.world_size == 4
        assert error.operation == "send_activation"
        assert "rank=1" in str(error)

    @pytest.mark.asyncio
    async def test_recv_failure_reports_node_identity(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=2, world_size=4)
        communicator.raise_on_recv = True

        with pytest.raises(PipelineNodeUnreachableError) as exc_info:
            await coordinator.forward_pipeline(torch.empty(0), "req-fail")

        error = exc_info.value
        assert error.rank == 1  # Previous rank
        assert error.world_size == 4
        assert error.operation == "recv_activation"

    @pytest.mark.asyncio
    async def test_error_includes_pipeline_position(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=1, world_size=3)
        communicator.raise_on_recv = True

        with pytest.raises(PipelineNodeUnreachableError) as exc_info:
            await coordinator.forward_pipeline(torch.empty(0), "req-fail")

        error_msg = str(exc_info.value)
        # Should include position description
        assert "rank=0" in error_msg
        assert "world_size=3" in error_msg

    @pytest.mark.asyncio
    async def test_generate_token_propagates_unreachable_error(self) -> None:
        coordinator, engine, communicator = _make_coordinator(rank=0, world_size=2)
        communicator.raise_on_send = True
        prompt = np.array([1, 2, 3], dtype=np.int64)

        with pytest.raises(PipelineNodeUnreachableError):
            await coordinator.generate_token(prompt)

    def test_describe_position_first(self) -> None:
        assert _describe_position(0, 4) == "first stage / embedding"

    def test_describe_position_last(self) -> None:
        assert _describe_position(3, 4) == "last stage / lm_head"

    def test_describe_position_middle(self) -> None:
        assert _describe_position(1, 4) == "middle stage 1/3"
        assert _describe_position(2, 4) == "middle stage 2/3"


# ---------------------------------------------------------------------------
# Tests: Variable world sizes (Requirement 5.8)
# ---------------------------------------------------------------------------


class TestVariableWorldSizes:
    """Tests that the coordinator supports 2, 3, and 4 node configurations."""

    @pytest.mark.asyncio
    async def test_world_size_2_full_pipeline(self) -> None:
        # Rank 0
        coord0, engine0, comm0 = _make_coordinator(rank=0, world_size=2, total_layers=10)
        input_tensor = torch.tensor([1], dtype=torch.long)
        result0 = await coord0.forward_pipeline(input_tensor, "req-ws2")
        assert result0 is None
        assert len(comm0.sent) == 1

        # Rank 1
        coord1, engine1, comm1 = _make_coordinator(rank=1, world_size=2, total_layers=10)
        comm1.queue_recv_tensor(torch.zeros(1, 64))
        result1 = await coord1.forward_pipeline(torch.empty(0), "req-ws2")
        assert result1 is not None

    @pytest.mark.asyncio
    async def test_world_size_3_middle_stage(self) -> None:
        coord, engine, comm = _make_coordinator(rank=1, world_size=3, total_layers=12)
        comm.queue_recv_tensor(torch.zeros(1, 64))
        result = await coord.forward_pipeline(torch.empty(0), "req-ws3")
        assert result is None
        assert len(comm.sent) == 1
        assert comm.sent[0].dst_rank == 2

    @pytest.mark.asyncio
    async def test_world_size_4_all_stages(self) -> None:
        # Verify all 4 stages can be created and have valid assignments
        for rank in range(4):
            coord, _, _ = _make_coordinator(rank=rank, world_size=4, total_layers=28)
            start, end = coord.get_stage_assignment()
            assert end > start
            assert start >= 0
            assert end <= 28
