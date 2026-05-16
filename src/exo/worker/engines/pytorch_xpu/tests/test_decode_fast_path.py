"""
Unit tests for decode fast-path send/receive functions.

Tests cover:
- send_decode_activation_fast rejects non-contiguous tensors
- send_decode_activation_fast rejects shape mismatches
- send_decode_activation_fast rejects dtype mismatches
- send_decode_activation_fast succeeds with valid tensor
- send_decode_activation_fast releases buffer on communication failure
- receive_decode_activation_fast succeeds and returns tensor on target device
- receive_decode_activation_fast releases buffer on communication failure

**Validates: Requirements 2.1, 2.3, 2.5, 2.12, 2.14**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"
_BUFFER_POOL_PATH = _THIS_DIR.parent / "buffer_pool.py"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module directly from file, avoiding __init__.py."""
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_distributed_mod = _load_module("distributed_fast_path_isolated", _DISTRIBUTED_PATH)
_buffer_pool_mod = _load_module("buffer_pool_fast_path_isolated", _BUFFER_POOL_PATH)

send_decode_activation_fast = _distributed_mod.send_decode_activation_fast
receive_decode_activation_fast = _distributed_mod.receive_decode_activation_fast
DecodeActivationProtocol = _distributed_mod.DecodeActivationProtocol
DecodeProtocolMismatchError = _distributed_mod.DecodeProtocolMismatchError
PipelineCommunicationError = _distributed_mod.PipelineCommunicationError
CommunicationBufferPool = _buffer_pool_mod.CommunicationBufferPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def protocol() -> DecodeActivationProtocol:
    """A standard decode activation protocol for testing."""
    return DecodeActivationProtocol(
        protocol_version=1,
        source_rank=0,
        destination_rank=1,
        dtype_name=str(torch.bfloat16),
        shape=(1, 1, 3584),
        maximum_microbatch_size=1,
        hidden_size=3584,
        requires_contiguous=True,
    )


@pytest.fixture
def buffer_pool() -> CommunicationBufferPool:
    """A fresh buffer pool for testing."""
    return CommunicationBufferPool()


@pytest.fixture
def mock_process_group() -> MagicMock:
    """A mock process group."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests: send_decode_activation_fast validation
# ---------------------------------------------------------------------------


class TestSendDecodeActivationFastValidation:
    """send_decode_activation_fast validates tensor properties.

    **Validates: Requirements 2.5, 2.12**
    """

    def test_rejects_non_contiguous_tensor(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Non-contiguous tensor raises DecodeProtocolMismatchError."""
        # Create a non-contiguous tensor by transposing
        tensor = torch.randn(3584, 1, 1, dtype=torch.bfloat16).permute(2, 1, 0)
        assert not tensor.is_contiguous()

        with pytest.raises(DecodeProtocolMismatchError, match="not contiguous") as exc_info:
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        error = exc_info.value
        assert error.source_rank == protocol.source_rank
        assert error.destination_rank == protocol.destination_rank

    def test_rejects_shape_mismatch(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Tensor with wrong shape raises DecodeProtocolMismatchError."""
        # Wrong shape: (1, 1, 4096) instead of (1, 1, 3584)
        tensor = torch.randn(1, 1, 4096, dtype=torch.bfloat16)

        with pytest.raises(DecodeProtocolMismatchError, match="does not match") as exc_info:
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        error = exc_info.value
        assert error.expected_shape == (1, 1, 3584)
        assert error.received_shape == (1, 1, 4096)
        assert error.source_rank == protocol.source_rank
        assert error.destination_rank == protocol.destination_rank

    def test_rejects_dtype_mismatch(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Tensor with wrong dtype raises DecodeProtocolMismatchError."""
        # Wrong dtype: float32 instead of bfloat16
        tensor = torch.randn(1, 1, 3584, dtype=torch.float32)

        with pytest.raises(DecodeProtocolMismatchError, match="dtype") as exc_info:
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        error = exc_info.value
        assert error.expected_dtype == str(torch.bfloat16)
        assert error.received_dtype == str(torch.float32)
        assert error.source_rank == protocol.source_rank
        assert error.destination_rank == protocol.destination_rank


# ---------------------------------------------------------------------------
# Tests: send_decode_activation_fast success path
# ---------------------------------------------------------------------------


class TestSendDecodeActivationFastSuccess:
    """send_decode_activation_fast succeeds with valid tensor.

    **Validates: Requirements 2.1, 2.3, 2.14**
    """

    def test_sends_valid_tensor(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Valid contiguous tensor with matching shape/dtype sends successfully."""
        tensor = torch.randn(1, 1, 3584, dtype=torch.bfloat16)

        with patch("torch.distributed.send") as mock_send:
            result = send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        assert result is None
        mock_send.assert_called_once()
        # Verify send was called with dst=1 (destination_rank)
        call_kwargs = mock_send.call_args
        assert call_kwargs.kwargs["dst"] == 1
        assert call_kwargs.kwargs["group"] is mock_process_group

    def test_buffer_released_after_success(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Buffer is released back to pool after successful send."""
        tensor = torch.randn(1, 1, 3584, dtype=torch.bfloat16)

        with patch("torch.distributed.send"):
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        # Buffer should be released (not active)
        stats = buffer_pool.get_statistics()
        assert stats.active_buffer_count == 0
        assert stats.total_buffer_count == 1  # Buffer was allocated


# ---------------------------------------------------------------------------
# Tests: send_decode_activation_fast error handling
# ---------------------------------------------------------------------------


class TestSendDecodeActivationFastErrors:
    """send_decode_activation_fast handles communication failures.

    **Validates: Requirements 2.12**
    """

    def test_releases_buffer_on_communication_failure(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Buffer is released even when dist.send raises."""
        tensor = torch.randn(1, 1, 3584, dtype=torch.bfloat16)

        with (
            patch("torch.distributed.send", side_effect=RuntimeError("Send failed")),
            pytest.raises(PipelineCommunicationError, match="send failed"),
        ):
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        # Buffer must still be released
        stats = buffer_pool.get_statistics()
        assert stats.active_buffer_count == 0


# ---------------------------------------------------------------------------
# Tests: receive_decode_activation_fast success path
# ---------------------------------------------------------------------------


class TestReceiveDecodeActivationFastSuccess:
    """receive_decode_activation_fast succeeds and returns tensor on target device.

    **Validates: Requirements 2.2, 2.3, 2.14**
    """

    def test_receives_and_returns_cpu_tensor(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Successful receive returns tensor with correct shape and dtype."""
        with patch("torch.distributed.recv") as mock_recv:
            # dist.recv fills the buffer in-place; we just let it pass
            result = receive_decode_activation_fast(
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
                target_device="cpu",
            )

        assert result.shape == (1, 1, 3584)
        assert result.dtype == torch.bfloat16
        mock_recv.assert_called_once()
        # Verify recv was called with src=0 (source_rank)
        call_kwargs = mock_recv.call_args
        assert call_kwargs.kwargs["src"] == 0
        assert call_kwargs.kwargs["group"] is mock_process_group

    def test_buffer_released_after_receive(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Buffer is released back to pool after successful receive."""
        with patch("torch.distributed.recv"):
            receive_decode_activation_fast(
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
                target_device="cpu",
            )

        stats = buffer_pool.get_statistics()
        assert stats.active_buffer_count == 0
        assert stats.total_buffer_count == 1


# ---------------------------------------------------------------------------
# Tests: receive_decode_activation_fast error handling
# ---------------------------------------------------------------------------


class TestReceiveDecodeActivationFastErrors:
    """receive_decode_activation_fast handles communication failures.

    **Validates: Requirements 2.12**
    """

    def test_releases_buffer_on_communication_failure(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Buffer is released even when dist.recv raises."""
        with (
            patch("torch.distributed.recv", side_effect=RuntimeError("Recv failed")),
            pytest.raises(PipelineCommunicationError, match="receive failed"),
        ):
            receive_decode_activation_fast(
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
                target_device="cpu",
            )

        # Buffer must still be released
        stats = buffer_pool.get_statistics()
        assert stats.active_buffer_count == 0


# ---------------------------------------------------------------------------
# Tests: buffer reuse across multiple calls
# ---------------------------------------------------------------------------


class TestBufferReuse:
    """Fast path reuses buffers across multiple decode steps.

    **Validates: Requirements 2.3, 2.4**
    """

    def test_send_reuses_buffer_on_second_call(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Second send call reuses the same buffer (no new allocation)."""
        tensor = torch.randn(1, 1, 3584, dtype=torch.bfloat16)

        with patch("torch.distributed.send"):
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )
            send_decode_activation_fast(
                activation=tensor,
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
            )

        stats = buffer_pool.get_statistics()
        assert stats.allocation_count == 1  # Only one allocation
        assert stats.reuse_count == 1  # Second call reused

    def test_receive_reuses_buffer_on_second_call(
        self,
        protocol: DecodeActivationProtocol,
        buffer_pool: CommunicationBufferPool,
        mock_process_group: MagicMock,
    ) -> None:
        """Second receive call reuses the same buffer (no new allocation)."""
        with patch("torch.distributed.recv"):
            receive_decode_activation_fast(
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
                target_device="cpu",
            )
            receive_decode_activation_fast(
                protocol=protocol,
                buffer_pool=buffer_pool,
                process_group=mock_process_group,
                target_device="cpu",
            )

        stats = buffer_pool.get_statistics()
        assert stats.allocation_count == 1  # Only one allocation
        assert stats.reuse_count == 1  # Second call reused
