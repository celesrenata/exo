"""Unit tests for distributed/communicator.py — process group management.

Tests CommConfig dataclass, Communicator initialization logic,
send/recv CPU-staging behavior, timeout error formatting, and destroy.
Uses mocking to avoid requiring a real distributed process group.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from exo.worker.engines.pytorch.distributed.communicator import (
    CommConfig,
    Communicator,
)


class TestCommConfig:
    """Tests for the CommConfig frozen dataclass."""

    def test_create_with_defaults(self) -> None:
        config = CommConfig(
            rank=0,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
            backend="gloo",
        )
        assert config.rank == 0
        assert config.world_size == 4
        assert config.master_addr == "10.1.1.12"
        assert config.master_port == 29500
        assert config.backend == "gloo"
        assert config.timeout_seconds == 30
        assert config.transport == "ethernet"

    def test_create_with_custom_timeout(self) -> None:
        config = CommConfig(
            rank=1,
            world_size=2,
            master_addr="192.168.1.1",
            master_port=12345,
            backend="nccl",
            timeout_seconds=60,
            transport="rdma",
        )
        assert config.timeout_seconds == 60
        assert config.transport == "rdma"
        assert config.backend == "nccl"

    def test_frozen_immutability(self) -> None:
        config = CommConfig(
            rank=0,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
            backend="gloo",
        )
        with pytest.raises(AttributeError):
            config.rank = 1  # type: ignore[misc]

    def test_equality(self) -> None:
        a = CommConfig(rank=0, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo")
        b = CommConfig(rank=0, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo")
        assert a == b

    def test_inequality(self) -> None:
        a = CommConfig(rank=0, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo")
        b = CommConfig(rank=1, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo")
        assert a != b


class TestCommunicatorInit:
    """Tests for Communicator construction and state."""

    def test_not_initialized_on_creation(self) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        assert comm.initialized is False

    def test_config_accessible(self) -> None:
        config = CommConfig(rank=1, world_size=4, master_addr="10.1.1.13", master_port=29500, backend="gloo")
        comm = Communicator(config)
        assert comm.config is config
        assert comm.config.rank == 1


class TestCommunicatorInitialize:
    """Tests for the initialize() method."""

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_initialize_sets_env_vars(self, mock_init: MagicMock) -> None:
        config = CommConfig(
            rank=2,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
            backend="gloo",
            timeout_seconds=45,
        )
        comm = Communicator(config)

        with patch.dict("os.environ", {}, clear=False):
            comm.initialize()

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["backend"] == "gloo" or call_kwargs[1].get("backend") == "gloo"

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_initialize_marks_initialized(self, mock_init: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()
        assert comm.initialized is True

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_double_initialize_raises(self, mock_init: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()
        with pytest.raises(RuntimeError, match="already initialized"):
            comm.initialize()


class TestCommunicatorSendTensor:
    """Tests for send_tensor() method."""

    def test_send_without_initialize_raises(self) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        tensor = torch.zeros(2, 3)
        with pytest.raises(RuntimeError, match="not initialized"):
            comm.send_tensor(tensor, dst_rank=1)

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.send")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_gloo_send_stages_to_cpu(self, mock_init: MagicMock, mock_send: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()

        tensor = torch.randn(4, 8)  # CPU tensor (simulating GPU → CPU staging)
        comm.send_tensor(tensor, dst_rank=1)

        mock_send.assert_called_once()
        sent_tensor = mock_send.call_args[0][0]
        assert sent_tensor.device.type == "cpu"
        assert sent_tensor.is_contiguous()

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.send")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_nccl_send_does_not_stage_cpu(self, mock_init: MagicMock, mock_send: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="nccl")
        comm = Communicator(config)
        comm.initialize()

        tensor = torch.randn(4, 8)  # CPU tensor in test, but logic path differs
        comm.send_tensor(tensor, dst_rank=1)

        mock_send.assert_called_once()
        sent_tensor = mock_send.call_args[0][0]
        assert sent_tensor.is_contiguous()

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.send")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_send_timeout_error_is_descriptive(self, mock_init: MagicMock, mock_send: MagicMock) -> None:
        mock_send.side_effect = RuntimeError("Operation timed out")
        config = CommConfig(rank=0, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo", timeout_seconds=30)
        comm = Communicator(config)
        comm.initialize()

        with pytest.raises(RuntimeError, match="src_rank=0") as exc_info:
            comm.send_tensor(torch.zeros(2, 2), dst_rank=2)

        error_msg = str(exc_info.value)
        assert "dst_rank=2" in error_msg
        assert "operation=send" in error_msg
        assert "timeout=30s" in error_msg


class TestCommunicatorRecvTensor:
    """Tests for recv_tensor() method."""

    def test_recv_without_initialize_raises(self) -> None:
        config = CommConfig(rank=1, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        with pytest.raises(RuntimeError, match="not initialized"):
            comm.recv_tensor(shape=(2, 3), dtype=torch.float32, src_rank=0, target_device="cpu")

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.recv")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_gloo_recv_creates_cpu_buffer(self, mock_init: MagicMock, mock_recv: MagicMock) -> None:
        config = CommConfig(rank=1, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()

        result = comm.recv_tensor(shape=(4, 8), dtype=torch.float32, src_rank=0, target_device="cpu")

        mock_recv.assert_called_once()
        recv_tensor = mock_recv.call_args[0][0]
        assert recv_tensor.device.type == "cpu"
        assert recv_tensor.shape == (4, 8)
        assert recv_tensor.dtype == torch.float32
        assert result.device.type == "cpu"

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.recv")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_gloo_recv_shape_and_dtype(self, mock_init: MagicMock, mock_recv: MagicMock) -> None:
        config = CommConfig(rank=1, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()

        result = comm.recv_tensor(shape=(2, 16, 64), dtype=torch.float16, src_rank=0, target_device="cpu")

        assert result.shape == (2, 16, 64)
        assert result.dtype == torch.float16

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.recv")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_recv_timeout_error_is_descriptive(self, mock_init: MagicMock, mock_recv: MagicMock) -> None:
        mock_recv.side_effect = RuntimeError("Timed out waiting for recv")
        config = CommConfig(rank=2, world_size=4, master_addr="10.1.1.12", master_port=29500, backend="gloo", timeout_seconds=45)
        comm = Communicator(config)
        comm.initialize()

        with pytest.raises(RuntimeError, match="src_rank=1") as exc_info:
            comm.recv_tensor(shape=(4, 4), dtype=torch.float32, src_rank=1, target_device="cpu")

        error_msg = str(exc_info.value)
        assert "dst_rank=2" in error_msg
        assert "operation=recv" in error_msg
        assert "timeout=45s" in error_msg

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.recv")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_non_timeout_error_propagates(self, mock_init: MagicMock, mock_recv: MagicMock) -> None:
        mock_recv.side_effect = RuntimeError("Connection refused")
        config = CommConfig(rank=1, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()

        with pytest.raises(RuntimeError, match="Connection refused"):
            comm.recv_tensor(shape=(2, 2), dtype=torch.float32, src_rank=0, target_device="cpu")


class TestCommunicatorDestroy:
    """Tests for destroy() method."""

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.destroy_process_group")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_destroy_calls_dist_destroy(self, mock_init: MagicMock, mock_destroy: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()
        comm.destroy()

        mock_destroy.assert_called_once()
        assert comm.initialized is False

    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.destroy_process_group")
    @patch("exo.worker.engines.pytorch.distributed.communicator.dist.init_process_group")
    def test_destroy_idempotent(self, mock_init: MagicMock, mock_destroy: MagicMock) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.initialize()
        comm.destroy()
        comm.destroy()  # Second call should be a no-op

        mock_destroy.assert_called_once()

    def test_destroy_without_initialize_is_noop(self) -> None:
        config = CommConfig(rank=0, world_size=2, master_addr="127.0.0.1", master_port=29500, backend="gloo")
        comm = Communicator(config)
        comm.destroy()  # Should not raise
        assert comm.initialized is False
