# Feature: pipeline-performance-optimization, Task 2: Token Result Point-to-Point
"""
Integration tests for token result point-to-point communication.

These tests require PyTorch to be installed (they use real torch.Tensor
operations for serialization round-trip verification). They are skipped
when torch is not available (e.g., on macOS dev machines).

**Validates: Requirements 2.7, 2.8, 2.9**
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Direct module import — bypass the heavy __init__.py import chain
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_DISTRIBUTED_PATH = _THIS_DIR.parent / "distributed.py"


def _load_distributed() -> types.ModuleType:
    """Load distributed.py directly from file, avoiding __init__.py."""
    module_name = "distributed_token_results_isolated"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _DISTRIBUTED_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_distributed()
TokenResultPacket = _mod.TokenResultPacket
send_token_results_to_rank_zero = _mod.send_token_results_to_rank_zero
receive_token_results_from_final_rank = _mod.receive_token_results_from_final_rank


class TestTokenResultSendReceiveRoundTrip:
    """End-to-end serialization round-trip for token result packets.

    These tests use real torch tensors to verify the full serialization
    and deserialization path works correctly.

    **Validates: Requirements 2.7, 2.8, 2.9**
    """

    def test_round_trip_preserves_all_fields(self) -> None:
        """Send followed by receive preserves all TokenResultPacket fields."""
        sent_tensors: list[torch.Tensor] = []

        mock_dist_send = MagicMock()
        mock_dist_send.get_rank.return_value = 3

        def capture_send(tensor, dst, group=None):  # noqa: ANN001, ANN201
            sent_tensors.append(tensor.clone())

        mock_dist_send.send = capture_send

        original_packet = TokenResultPacket(
            request_identifier="round-trip-test-123",
            token_identifier=54321,
            position=42,
            finished=True,
            finish_reason="length",
        )

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_send}):
            send_token_results_to_rank_zero(
                packet=original_packet,
                process_group=MagicMock(),
                performance_recorder=None,
            )

        # Now simulate receive using the captured tensors
        header_tensor = sent_tensors[0]
        payload_tensor = sent_tensors[1]

        mock_dist_recv = MagicMock()
        recv_call_count = [0]

        def fill_from_captured(buffer, src, group=None):  # noqa: ANN001, ANN201
            if recv_call_count[0] == 0:
                buffer.copy_(header_tensor)
            else:
                buffer.copy_(payload_tensor)
            recv_call_count[0] += 1

        mock_dist_recv.recv = fill_from_captured

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_recv}):
            received = receive_token_results_from_final_rank(
                process_group=MagicMock(),
                world_size=4,
                performance_recorder=None,
            )

        assert received.request_identifier == original_packet.request_identifier
        assert received.token_identifier == original_packet.token_identifier
        assert received.position == original_packet.position
        assert received.finished == original_packet.finished
        assert received.finish_reason == original_packet.finish_reason

    def test_round_trip_with_empty_request_id(self) -> None:
        """Round-trip works with an empty request_identifier."""
        sent_tensors: list[torch.Tensor] = []

        mock_dist_send = MagicMock()
        mock_dist_send.get_rank.return_value = 3

        def capture_send(tensor, dst, group=None):  # noqa: ANN001, ANN201
            sent_tensors.append(tensor.clone())

        mock_dist_send.send = capture_send

        original_packet = TokenResultPacket(
            request_identifier="",
            token_identifier=0,
            position=0,
            finished=False,
            finish_reason=None,
        )

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_send}):
            send_token_results_to_rank_zero(
                packet=original_packet,
                process_group=MagicMock(),
                performance_recorder=None,
            )

        header_tensor = sent_tensors[0]
        payload_tensor = sent_tensors[1]

        mock_dist_recv = MagicMock()
        recv_call_count = [0]

        def fill_from_captured(buffer, src, group=None):  # noqa: ANN001, ANN201
            if recv_call_count[0] == 0:
                buffer.copy_(header_tensor)
            else:
                buffer.copy_(payload_tensor)
            recv_call_count[0] += 1

        mock_dist_recv.recv = fill_from_captured

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_recv}):
            received = receive_token_results_from_final_rank(
                process_group=MagicMock(),
                world_size=4,
                performance_recorder=None,
            )

        assert received.request_identifier == ""
        assert received.token_identifier == 0
        assert received.position == 0
        assert received.finished is False
        assert received.finish_reason is None

    def test_round_trip_with_unicode_request_id(self) -> None:
        """Round-trip works with a Unicode request_identifier."""
        sent_tensors: list[torch.Tensor] = []

        mock_dist_send = MagicMock()
        mock_dist_send.get_rank.return_value = 3

        def capture_send(tensor, dst, group=None):  # noqa: ANN001, ANN201
            sent_tensors.append(tensor.clone())

        mock_dist_send.send = capture_send

        original_packet = TokenResultPacket(
            request_identifier="req-日本語-テスト",
            token_identifier=12345,
            position=99,
            finished=True,
            finish_reason="stop",
        )

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_send}):
            send_token_results_to_rank_zero(
                packet=original_packet,
                process_group=MagicMock(),
                performance_recorder=None,
            )

        header_tensor = sent_tensors[0]
        payload_tensor = sent_tensors[1]

        mock_dist_recv = MagicMock()
        recv_call_count = [0]

        def fill_from_captured(buffer, src, group=None):  # noqa: ANN001, ANN201
            if recv_call_count[0] == 0:
                buffer.copy_(header_tensor)
            else:
                buffer.copy_(payload_tensor)
            recv_call_count[0] += 1

        mock_dist_recv.recv = fill_from_captured

        with patch.dict(sys.modules, {"torch.distributed": mock_dist_recv}):
            received = receive_token_results_from_final_rank(
                process_group=MagicMock(),
                world_size=4,
                performance_recorder=None,
            )

        assert received.request_identifier == "req-日本語-テスト"
        assert received.token_identifier == 12345
        assert received.position == 99
        assert received.finished is True
        assert received.finish_reason == "stop"

    def test_send_produces_two_tensors(self) -> None:
        """send_token_results_to_rank_zero sends exactly two tensors."""
        sent_tensors: list[torch.Tensor] = []

        mock_dist = MagicMock()
        mock_dist.get_rank.return_value = 3

        def capture_send(tensor, dst, group=None):  # noqa: ANN001, ANN201
            sent_tensors.append(tensor.clone())

        mock_dist.send = capture_send

        packet = TokenResultPacket(
            request_identifier="req-001",
            token_identifier=42,
            position=5,
            finished=False,
            finish_reason=None,
        )

        with patch.dict(sys.modules, {"torch.distributed": mock_dist}):
            send_token_results_to_rank_zero(
                packet=packet,
                process_group=MagicMock(),
                performance_recorder=None,
            )

        assert len(sent_tensors) == 2
        # Header: int64, shape [5]
        assert sent_tensors[0].dtype == torch.int64
        assert sent_tensors[0].shape == (5,)
        # Payload: uint8, shape [256]
        assert sent_tensors[1].dtype == torch.uint8
        assert sent_tensors[1].shape == (256,)

    def test_header_contains_correct_values(self) -> None:
        """Header tensor encodes token_id, position, finished, reason, id_len."""
        sent_tensors: list[torch.Tensor] = []

        mock_dist = MagicMock()
        mock_dist.get_rank.return_value = 3

        def capture_send(tensor, dst, group=None):  # noqa: ANN001, ANN201
            sent_tensors.append(tensor.clone())

        mock_dist.send = capture_send

        packet = TokenResultPacket(
            request_identifier="test-req",
            token_identifier=1234,
            position=10,
            finished=True,
            finish_reason="stop",
        )

        with patch.dict(sys.modules, {"torch.distributed": mock_dist}):
            send_token_results_to_rank_zero(
                packet=packet,
                process_group=MagicMock(),
                performance_recorder=None,
            )

        header = sent_tensors[0]
        assert header[0].item() == 1234  # token_identifier
        assert header[1].item() == 10  # position
        assert header[2].item() == 1  # finished = True
        assert header[3].item() == 1  # finish_reason = "stop" -> 1
        assert header[4].item() == len("test-req".encode("utf-8"))
