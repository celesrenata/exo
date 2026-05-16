"""
Unit tests for decode fast-path protocol negotiation.

Tests cover:
- Final rank returns None (no downstream neighbor)
- Rank 0 respond returns None (no upstream neighbor)
- Successful negotiation produces valid DecodeActivationProtocol
- Protocol version mismatch raises DecodeProtocolMismatchError
- Hidden size mismatch raises DecodeProtocolMismatchError
- Dtype mismatch raises DecodeProtocolMismatchError
- Communication failure raises PipelineCommunicationError
- Unsupported dtype raises DecodeProtocolMismatchError
- Agreed microbatch size uses minimum of both sides

**Validates: Requirements 2.1, 2.2, 2.5, 2.11**
"""

from __future__ import annotations

import importlib.util
import sys
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
    module_name = "distributed_negotiation_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
negotiate_decode_activation_protocol = _mod.negotiate_decode_activation_protocol
respond_to_decode_protocol_negotiation = _mod.respond_to_decode_protocol_negotiation
DecodeActivationProtocol = _mod.DecodeActivationProtocol
DecodeProtocolMismatchError = _mod.DecodeProtocolMismatchError
PipelineCommunicationError = _mod.PipelineCommunicationError
DECODE_FAST_PATH_PROTOCOL_VERSION = _mod.DECODE_FAST_PATH_PROTOCOL_VERSION
_NEGOTIATION_METADATA_LENGTH = _mod._NEGOTIATION_METADATA_LENGTH


# ---------------------------------------------------------------------------
# Mock helpers — no real torch required
# ---------------------------------------------------------------------------


class MockDtype:
    """Mock for torch.dtype with string representation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __str__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MockDtype):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


BFLOAT16 = MockDtype("bfloat16")
FLOAT16 = MockDtype("float16")
FLOAT32 = MockDtype("float32")
INT8 = MockDtype("int8")


class MockTensor:
    """Mock tensor that supports indexing and item() for metadata exchange."""

    def __init__(self, data: list[int]) -> None:
        self._data = list(data)

    def __getitem__(self, idx: int) -> "MockTensorElement":
        return MockTensorElement(self._data[idx])

    def __setitem__(self, idx: int, value: int) -> None:
        self._data[idx] = value

    def tolist(self) -> list[int]:
        return list(self._data)


class MockTensorElement:
    """Mock tensor element that supports .item()."""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


def _make_torch_mock_with_tensors(
    remote_metadata_values: list[int] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create torch and torch.distributed mocks with tensor creation support.

    Args:
        remote_metadata_values: If provided, the recv mock will populate the
            buffer with these values to simulate the remote rank's response.
    """
    mock_dist = MagicMock()
    mock_torch = MagicMock()
    mock_torch.distributed = mock_dist

    # Set up dtype constants
    mock_torch.int64 = "int64"
    mock_torch.float16 = FLOAT16
    mock_torch.bfloat16 = BFLOAT16
    mock_torch.float32 = FLOAT32
    mock_torch.float64 = MockDtype("float64")
    mock_torch.int8 = INT8

    # Track created tensors so we can inspect them
    created_tensors: list[MockTensor] = []

    def mock_tensor_constructor(data: list[int], dtype: str = "int64", device: str = "cpu") -> MockTensor:
        t = MockTensor(data)
        created_tensors.append(t)
        return t

    def mock_empty_constructor(size: int, dtype: str = "int64", device: str = "cpu") -> MockTensor:
        t = MockTensor([0] * size)
        created_tensors.append(t)
        return t

    mock_torch.tensor = mock_tensor_constructor
    mock_torch.empty = mock_empty_constructor

    # Set up recv to populate the buffer with remote metadata
    if remote_metadata_values is not None:
        def mock_recv(tensor: MockTensor, src: int, group: object = None) -> None:
            for i, val in enumerate(remote_metadata_values):
                tensor[i] = val

        mock_dist.recv.side_effect = mock_recv

    return mock_torch, mock_dist


def _patch_torch(mock_torch: MagicMock, mock_dist: MagicMock):  # noqa: ANN201
    """Context manager to patch torch modules for negotiation tests."""
    return patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.distributed": mock_dist,
    })


# ---------------------------------------------------------------------------
# Tests: Final rank returns None
# ---------------------------------------------------------------------------


class TestFinalRankReturnsNone:
    """Final rank (world_size - 1) has no downstream neighbor.

    **Validates: Requirements 2.1, 2.11**
    """

    def test_final_rank_returns_none(self) -> None:
        """negotiate_decode_activation_protocol returns None for the final rank."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=3,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        assert result is None

    def test_final_rank_does_not_send_or_receive(self) -> None:
        """Final rank does not call dist.send or dist.recv."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=3,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        mock_dist.send.assert_not_called()
        mock_dist.recv.assert_not_called()

    def test_world_size_two_final_rank(self) -> None:
        """Final rank in a 2-rank world also returns None."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=1,
                world_size=2,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        assert result is None


# ---------------------------------------------------------------------------
# Tests: Rank 0 respond returns None
# ---------------------------------------------------------------------------


class TestRankZeroRespondReturnsNone:
    """Rank 0 has no upstream neighbor for respond.

    **Validates: Requirements 2.1, 2.11**
    """

    def test_rank_zero_respond_returns_none(self) -> None:
        """respond_to_decode_protocol_negotiation returns None for rank 0."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = respond_to_decode_protocol_negotiation(
                process_group=mock_process_group,
                local_rank=0,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        assert result is None

    def test_rank_zero_respond_does_not_communicate(self) -> None:
        """Rank 0 respond does not call dist.send or dist.recv."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            respond_to_decode_protocol_negotiation(
                process_group=mock_process_group,
                local_rank=0,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        mock_dist.send.assert_not_called()
        mock_dist.recv.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Successful negotiation
# ---------------------------------------------------------------------------


class TestSuccessfulNegotiation:
    """Successful protocol negotiation produces a valid DecodeActivationProtocol.

    **Validates: Requirements 2.1, 2.2, 2.5, 2.11**
    """

    def test_negotiate_returns_protocol_on_matching_metadata(self) -> None:
        """Matching metadata from downstream produces a valid protocol."""
        # Remote responds with: version=1, hidden=3584, dtype=bfloat16(2), microbatch=1
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=0,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        assert result is not None
        assert isinstance(result, DecodeActivationProtocol)
        assert result.protocol_version == DECODE_FAST_PATH_PROTOCOL_VERSION
        assert result.source_rank == 0
        assert result.destination_rank == 1
        assert result.hidden_size == 3584
        assert result.dtype_name == str(BFLOAT16)
        assert result.maximum_microbatch_size == 1
        assert result.shape == (1, 1, 3584)
        assert result.requires_contiguous is True

    def test_negotiate_uses_minimum_microbatch_size(self) -> None:
        """Protocol uses the minimum of both sides' maximum microbatch size."""
        # Remote has larger microbatch size (4), local has 2
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 4]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=1,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=2,
            )

        assert result is not None
        assert result.maximum_microbatch_size == 2  # min(2, 4) = 2
        assert result.shape == (2, 1, 3584)

    def test_negotiate_sends_to_downstream_rank(self) -> None:
        """negotiate sends metadata to rank+1."""
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=1,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        # Verify send was called with dst=2 (rank+1)
        send_call = mock_dist.send.call_args
        assert send_call.kwargs.get("dst") == 2 or send_call[1].get("dst") == 2

    def test_negotiate_receives_from_downstream_rank(self) -> None:
        """negotiate receives metadata from rank+1."""
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            negotiate_decode_activation_protocol(
                process_group=mock_process_group,
                local_rank=0,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        # Verify recv was called with src=1 (rank+1)
        recv_call = mock_dist.recv.call_args
        assert recv_call.kwargs.get("src") == 1 or recv_call[1].get("src") == 1


# ---------------------------------------------------------------------------
# Tests: Protocol version mismatch
# ---------------------------------------------------------------------------


class TestProtocolVersionMismatch:
    """Protocol version mismatch raises DecodeProtocolMismatchError.

    **Validates: Requirements 2.5, 2.12**
    """

    def test_version_mismatch_raises_error(self) -> None:
        """Mismatched protocol version raises DecodeProtocolMismatchError."""
        # Remote has wrong protocol version (99)
        remote_values = [99, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Protocol version mismatch" in str(error)
        assert error.source_rank == 0
        assert error.destination_rank == 1
        assert error.protocol_version == DECODE_FAST_PATH_PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# Tests: Hidden size mismatch
# ---------------------------------------------------------------------------


class TestHiddenSizeMismatch:
    """Hidden size mismatch raises DecodeProtocolMismatchError.

    **Validates: Requirements 2.5, 2.12**
    """

    def test_hidden_size_mismatch_raises_error(self) -> None:
        """Mismatched hidden size raises DecodeProtocolMismatchError."""
        # Remote has different hidden size (4096 vs local 3584)
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 4096, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Hidden size mismatch" in str(error)
        assert error.source_rank == 0
        assert error.destination_rank == 1
        assert error.expected_shape is not None
        assert error.received_shape is not None


# ---------------------------------------------------------------------------
# Tests: Dtype mismatch
# ---------------------------------------------------------------------------


class TestDtypeMismatch:
    """Dtype mismatch raises DecodeProtocolMismatchError.

    **Validates: Requirements 2.5, 2.12**
    """

    def test_dtype_mismatch_raises_error(self) -> None:
        """Mismatched dtype raises DecodeProtocolMismatchError."""
        # Remote has float16 (encoding 1), local has bfloat16 (encoding 2)
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 1, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Dtype mismatch" in str(error)
        assert error.source_rank == 0
        assert error.destination_rank == 1
        assert error.expected_dtype is not None
        assert error.received_dtype is not None

    def test_unknown_remote_dtype_encoding(self) -> None:
        """Unknown remote dtype encoding produces descriptive error."""
        # Remote has unknown dtype encoding (99)
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 99, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Dtype mismatch" in str(error)
        assert "unknown" in str(error.received_dtype)


# ---------------------------------------------------------------------------
# Tests: Communication failure
# ---------------------------------------------------------------------------


class TestCommunicationFailure:
    """Communication failure raises PipelineCommunicationError.

    **Validates: Requirements 2.12**
    """

    def test_send_failure_raises_pipeline_error(self) -> None:
        """Failed send raises PipelineCommunicationError."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_dist.send.side_effect = RuntimeError("Connection refused")
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(PipelineCommunicationError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert error.source_rank == 0
        assert error.destination_rank == 1

    def test_recv_failure_raises_pipeline_error(self) -> None:
        """Failed recv raises PipelineCommunicationError."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_dist.send.return_value = None
        mock_dist.recv.side_effect = RuntimeError("Peer disconnected")
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(PipelineCommunicationError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert error.source_rank == 1  # recv failure: source is downstream
        assert error.destination_rank == 0

    def test_respond_recv_failure_raises_pipeline_error(self) -> None:
        """Failed recv in respond raises PipelineCommunicationError."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_dist.recv.side_effect = RuntimeError("Upstream disconnected")
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(PipelineCommunicationError) as exc_info:
                respond_to_decode_protocol_negotiation(
                    process_group=mock_process_group,
                    local_rank=1,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert error.source_rank == 0  # upstream
        assert error.destination_rank == 1


# ---------------------------------------------------------------------------
# Tests: Unsupported dtype
# ---------------------------------------------------------------------------


class TestUnsupportedDtype:
    """Unsupported dtype raises DecodeProtocolMismatchError.

    **Validates: Requirements 2.5**
    """

    def test_unsupported_dtype_raises_error(self) -> None:
        """Unsupported dtype raises DecodeProtocolMismatchError before communication."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                negotiate_decode_activation_protocol(
                    process_group=mock_process_group,
                    local_rank=0,
                    world_size=4,
                    hidden_size=3584,
                    dtype=INT8,  # Not in the encoding map
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Unsupported dtype" in str(error)

        # Should not have attempted communication
        mock_dist.send.assert_not_called()
        mock_dist.recv.assert_not_called()

    def test_respond_unsupported_dtype_raises_error(self) -> None:
        """Unsupported dtype in respond raises DecodeProtocolMismatchError."""
        mock_torch, mock_dist = _make_torch_mock_with_tensors()
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                respond_to_decode_protocol_negotiation(
                    process_group=mock_process_group,
                    local_rank=1,
                    world_size=4,
                    hidden_size=3584,
                    dtype=INT8,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Unsupported dtype" in str(error)


# ---------------------------------------------------------------------------
# Tests: Respond to negotiation
# ---------------------------------------------------------------------------


class TestRespondToNegotiation:
    """respond_to_decode_protocol_negotiation handles the receiving side.

    **Validates: Requirements 2.1, 2.2, 2.5, 2.11**
    """

    def test_respond_returns_protocol_on_matching_metadata(self) -> None:
        """Matching metadata from upstream produces a valid protocol."""
        # Upstream sends: version=1, hidden=3584, dtype=bfloat16(2), microbatch=1
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            result = respond_to_decode_protocol_negotiation(
                process_group=mock_process_group,
                local_rank=1,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        assert result is not None
        assert isinstance(result, DecodeActivationProtocol)
        assert result.protocol_version == DECODE_FAST_PATH_PROTOCOL_VERSION
        assert result.source_rank == 0  # upstream
        assert result.destination_rank == 1  # local
        assert result.hidden_size == 3584
        assert result.dtype_name == str(BFLOAT16)
        assert result.maximum_microbatch_size == 1
        assert result.shape == (1, 1, 3584)

    def test_respond_version_mismatch_raises_error(self) -> None:
        """Version mismatch in respond raises DecodeProtocolMismatchError."""
        # Upstream sends wrong version
        remote_values = [42, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            with pytest.raises(DecodeProtocolMismatchError) as exc_info:
                respond_to_decode_protocol_negotiation(
                    process_group=mock_process_group,
                    local_rank=1,
                    world_size=4,
                    hidden_size=3584,
                    dtype=BFLOAT16,
                    maximum_microbatch_size=1,
                )

        error = exc_info.value
        assert "Protocol version mismatch" in str(error)
        assert error.source_rank == 0
        assert error.destination_rank == 1

    def test_respond_receives_from_upstream_sends_to_upstream(self) -> None:
        """respond receives from rank-1 and sends back to rank-1."""
        remote_values = [DECODE_FAST_PATH_PROTOCOL_VERSION, 3584, 2, 1]
        mock_torch, mock_dist = _make_torch_mock_with_tensors(remote_values)
        mock_process_group = MagicMock()

        with _patch_torch(mock_torch, mock_dist):
            respond_to_decode_protocol_negotiation(
                process_group=mock_process_group,
                local_rank=2,
                world_size=4,
                hidden_size=3584,
                dtype=BFLOAT16,
                maximum_microbatch_size=1,
            )

        # Verify recv was called with src=1 (upstream)
        recv_call = mock_dist.recv.call_args
        assert recv_call.kwargs.get("src") == 1 or recv_call[1].get("src") == 1

        # Verify send was called with dst=1 (upstream)
        send_call = mock_dist.send.call_args
        assert send_call.kwargs.get("dst") == 1 or send_call[1].get("dst") == 1


# ---------------------------------------------------------------------------
# Tests: Protocol version constant
# ---------------------------------------------------------------------------


class TestProtocolVersionConstant:
    """DECODE_FAST_PATH_PROTOCOL_VERSION is a positive integer.

    **Validates: Requirements 2.1**
    """

    def test_protocol_version_is_positive_integer(self) -> None:
        """Protocol version constant is a positive integer."""
        assert isinstance(DECODE_FAST_PATH_PROTOCOL_VERSION, int)
        assert DECODE_FAST_PATH_PROTOCOL_VERSION > 0

    def test_metadata_length_is_four(self) -> None:
        """Negotiation metadata tensor has 4 elements."""
        assert _NEGOTIATION_METADATA_LENGTH == 4
