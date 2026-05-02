# Feature: distributed-gpu-sharding, Task 3.5: Distributed communicator unit tests
"""
Unit tests for the distributed communicator module.

Tests cover:
- init_process_group success → no exception
- init_process_group timeout → error message contains rank, world_size, "gloo"
- destroy_process_group timeout → warning logged, no exception raised
- send_activation failure → error message contains src rank, dst rank, tensor shape
- recv_activation failure → error message contains src rank, local rank, tensor shape

**Validates: Requirements 1.4, 1.5, 1.6, 8.1, 8.2, 8.4**
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"


def _load_distributed() -> types.ModuleType:
    """Load distributed.py directly from file, avoiding __init__.py."""
    module_name = "distributed_unit_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
ProcessGroupConfig = _mod.ProcessGroupConfig
CpuStagedTensor = _mod.CpuStagedTensor
init_process_group = _mod.init_process_group
destroy_process_group = _mod.destroy_process_group
send_activation = _mod.send_activation
recv_activation = _mod.recv_activation
stage_to_cpu = _mod.stage_to_cpu
unstage_from_cpu = _mod.unstage_from_cpu


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_torch_mocks() -> tuple[MagicMock, MagicMock]:
    """Create paired torch and torch.distributed mocks.

    Returns (mock_torch, mock_dist) where mock_torch.distributed IS mock_dist,
    ensuring that both `sys.modules["torch.distributed"]` and
    `sys.modules["torch"].distributed` resolve to the same object.
    """
    mock_dist = MagicMock()
    mock_torch = MagicMock()
    mock_torch.distributed = mock_dist
    return mock_torch, mock_dist


def _patch_torch(mock_torch: MagicMock, mock_dist: MagicMock):  # noqa: ANN201
    """Context manager to patch torch and torch.distributed in sys.modules."""
    return patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.distributed": mock_dist,
    })


class MockDtype:
    """Mock for torch.dtype."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MockDtype):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


FLOAT32 = MockDtype("float32")
FLOAT16 = MockDtype("float16")


class MockTensor:
    """Mock tensor that behaves like torch.Tensor for staging/send/recv tests."""

    def __init__(self, shape: tuple[int, ...], dtype: MockDtype, device: str) -> None:
        self._shape = shape
        self._dtype = dtype
        self._device = device

    @property
    def dtype(self) -> MockDtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> str:
        return self._device

    def to(self, target: str) -> "MockTensor":
        return MockTensor(shape=self._shape, dtype=self._dtype, device=target)


# ---------------------------------------------------------------------------
# Tests: init_process_group success → no exception
# ---------------------------------------------------------------------------


class TestInitProcessGroupSuccess:
    """init_process_group with successful torch.distributed call.

    **Validates: Requirements 1.4, 1.6**
    """

    def test_init_success_no_exception(self) -> None:
        """Successful init_process_group does not raise."""
        config = ProcessGroupConfig(
            rank=0,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
        )

        mock_torch, mock_dist = _make_torch_mocks()

        orig_addr = os.environ.get("MASTER_ADDR")
        orig_port = os.environ.get("MASTER_PORT")

        try:
            with _patch_torch(mock_torch, mock_dist):
                init_process_group(config)

            # Verify it was called
            mock_dist.init_process_group.assert_called_once()

            # Verify env vars were set
            assert os.environ["MASTER_ADDR"] == "10.1.1.12"
            assert os.environ["MASTER_PORT"] == "29500"
        finally:
            if orig_addr is not None:
                os.environ["MASTER_ADDR"] = orig_addr
            elif "MASTER_ADDR" in os.environ:
                del os.environ["MASTER_ADDR"]
            if orig_port is not None:
                os.environ["MASTER_PORT"] = orig_port
            elif "MASTER_PORT" in os.environ:
                del os.environ["MASTER_PORT"]

    def test_init_uses_gloo_backend(self) -> None:
        """init_process_group always uses backend='gloo'."""
        config = ProcessGroupConfig(
            rank=1,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
        )

        mock_torch, mock_dist = _make_torch_mocks()

        orig_addr = os.environ.get("MASTER_ADDR")
        orig_port = os.environ.get("MASTER_PORT")

        try:
            with _patch_torch(mock_torch, mock_dist):
                init_process_group(config)

            call_kwargs = mock_dist.init_process_group.call_args.kwargs
            assert call_kwargs["backend"] == "gloo"
            assert call_kwargs["init_method"] == "env://"
        finally:
            if orig_addr is not None:
                os.environ["MASTER_ADDR"] = orig_addr
            elif "MASTER_ADDR" in os.environ:
                del os.environ["MASTER_ADDR"]
            if orig_port is not None:
                os.environ["MASTER_PORT"] = orig_port
            elif "MASTER_PORT" in os.environ:
                del os.environ["MASTER_PORT"]


# ---------------------------------------------------------------------------
# Tests: init_process_group timeout → error message contains rank, world_size, "gloo"
# ---------------------------------------------------------------------------


class TestInitProcessGroupFailure:
    """init_process_group failure produces descriptive error messages.

    **Validates: Requirements 1.5, 1.6**
    """

    def test_init_timeout_error_contains_rank_worldsize_gloo(self) -> None:
        """Timeout error message contains rank, world_size, and 'gloo'."""
        config = ProcessGroupConfig(
            rank=2,
            world_size=4,
            master_addr="10.1.1.12",
            master_port=29500,
        )

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.init_process_group.side_effect = RuntimeError(
            "Connection timed out after 120 seconds"
        )

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                init_process_group(config)

        error_msg = str(exc_info.value)
        assert "rank=2" in error_msg
        assert "world_size=4" in error_msg
        assert "gloo" in error_msg

    def test_init_failure_preserves_original_exception(self) -> None:
        """The original exception is chained via __cause__."""
        config = ProcessGroupConfig(
            rank=0,
            world_size=2,
            master_addr="10.1.1.12",
            master_port=29500,
        )

        original_error = RuntimeError("Peer unreachable")
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.init_process_group.side_effect = original_error

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                init_process_group(config)

        assert exc_info.value.__cause__ is original_error


# ---------------------------------------------------------------------------
# Tests: destroy_process_group timeout → warning logged, no exception raised
# ---------------------------------------------------------------------------


class TestDestroyProcessGroup:
    """destroy_process_group handles timeout gracefully.

    **Validates: Requirements 8.1, 8.2, 8.4**
    """

    def test_destroy_success_no_exception(self) -> None:
        """Successful destroy_process_group does not raise."""
        mock_torch, mock_dist = _make_torch_mocks()

        with _patch_torch(mock_torch, mock_dist):
            destroy_process_group(timeout_seconds=5.0)

        mock_dist.destroy_process_group.assert_called_once()

    def test_destroy_timeout_logs_warning_no_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Timeout during destroy logs a warning but does not raise.

        **Validates: Requirements 8.2, 8.4**
        """
        mock_torch, mock_dist = _make_torch_mocks()

        def slow_destroy() -> None:
            time.sleep(10)  # Longer than timeout

        mock_dist.destroy_process_group.side_effect = slow_destroy

        with (
            _patch_torch(mock_torch, mock_dist),
            caplog.at_level(logging.WARNING),
        ):
            # Use a very short timeout so the test runs quickly
            destroy_process_group(timeout_seconds=0.1)

        # Should not raise, and should log a warning
        assert any(
            "timed out" in record.message
            for record in caplog.records
        ), f"Expected timeout warning, got: {[r.message for r in caplog.records]}"

    def test_destroy_exception_logs_warning_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception during destroy logs a warning but does not raise.

        **Validates: Requirement 8.4**
        """
        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.destroy_process_group.side_effect = RuntimeError("Peer gone")

        with (
            _patch_torch(mock_torch, mock_dist),
            caplog.at_level(logging.WARNING),
        ):
            destroy_process_group(timeout_seconds=5.0)

        assert any(
            "exception" in record.message.lower()
            for record in caplog.records
        ), f"Expected exception warning, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Tests: send_activation failure → error message contains src rank, dst rank, tensor shape
# ---------------------------------------------------------------------------


class TestSendActivationFailure:
    """send_activation failure produces descriptive error messages.

    **Validates: Requirements 1.5, 2.6**
    """

    def test_send_failure_contains_ranks_and_shape(self) -> None:
        """Send failure error message contains src_rank, dst_rank, and tensor_shape."""
        tensor = MockTensor(shape=(2, 4096), dtype=FLOAT16, device="cuda:0")

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.send.side_effect = RuntimeError("Send timed out")
        mock_dist.get_rank.return_value = 0

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                send_activation(tensor, dst_rank=1)

        error_msg = str(exc_info.value)
        assert "src_rank=0" in error_msg
        assert "dst_rank=1" in error_msg
        assert "(2, 4096)" in error_msg

    def test_send_failure_with_different_ranks(self) -> None:
        """Send failure with non-adjacent ranks still reports correctly."""
        tensor = MockTensor(shape=(1, 8, 512), dtype=FLOAT32, device="xpu:0")

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.send.side_effect = RuntimeError("Peer disconnected")
        mock_dist.get_rank.return_value = 2

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                send_activation(tensor, dst_rank=3)

        error_msg = str(exc_info.value)
        assert "src_rank=2" in error_msg
        assert "dst_rank=3" in error_msg
        assert "(1, 8, 512)" in error_msg


# ---------------------------------------------------------------------------
# Tests: recv_activation failure → error message contains src rank, local rank, tensor shape
# ---------------------------------------------------------------------------


class TestRecvActivationFailure:
    """recv_activation failure produces descriptive error messages.

    **Validates: Requirements 1.5, 2.6**
    """

    def test_recv_failure_contains_ranks_and_shape(self) -> None:
        """Recv failure error message contains src_rank, local_rank, and tensor_shape."""
        shape = (2, 4096)

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.recv.side_effect = RuntimeError("Recv timed out")
        mock_dist.get_rank.return_value = 1

        mock_buffer = MockTensor(shape=shape, dtype=FLOAT16, device="cpu")
        mock_torch.empty.return_value = mock_buffer

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                recv_activation(shape, FLOAT16, src_rank=0, target_device="xpu:0")

        error_msg = str(exc_info.value)
        assert "src_rank=0" in error_msg
        assert "local_rank=1" in error_msg
        assert "(2, 4096)" in error_msg

    def test_recv_failure_with_different_ranks(self) -> None:
        """Recv failure with different rank combination reports correctly."""
        shape = (4, 16, 256)

        mock_torch, mock_dist = _make_torch_mocks()
        mock_dist.recv.side_effect = RuntimeError("Peer disconnected")
        mock_dist.get_rank.return_value = 3

        mock_buffer = MockTensor(shape=shape, dtype=FLOAT32, device="cpu")
        mock_torch.empty.return_value = mock_buffer

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(RuntimeError) as exc_info:
                recv_activation(shape, FLOAT32, src_rank=2, target_device="cuda:0")

        error_msg = str(exc_info.value)
        assert "src_rank=2" in error_msg
        assert "local_rank=3" in error_msg
        assert "(4, 16, 256)" in error_msg
