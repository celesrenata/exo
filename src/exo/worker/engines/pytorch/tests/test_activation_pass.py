"""Unit tests for pipeline/activation_pass.py — inter-stage activation transfer.

Tests send_activation() and recv_activation() with a mock communicator,
verifying metadata correctness, dtype handling, and logging behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from exo.worker.engines.pytorch.pipeline.activation_pass import (
    ActivationMessage,
    recv_activation,
    send_activation,
)


# ---------------------------------------------------------------------------
# Mock communicator
# ---------------------------------------------------------------------------


@dataclass
class MockCommunicator:
    """Records send/recv calls for verification."""

    sent_tensors: list[tuple[torch.Tensor, int]] = field(default_factory=list)
    recv_return: torch.Tensor | None = None

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self.sent_tensors.append((tensor, dst_rank))

    def recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        target_device: str,
    ) -> torch.Tensor:
        if self.recv_return is not None:
            return self.recv_return
        # Return a zeros tensor with the requested shape/dtype on CPU
        return torch.zeros(shape, dtype=dtype, device="cpu")


# ---------------------------------------------------------------------------
# Tests for send_activation
# ---------------------------------------------------------------------------


class TestSendActivation:
    """Tests for send_activation()."""

    def test_sends_tensor_to_correct_rank(self) -> None:
        comm = MockCommunicator()
        tensor = torch.randn(2, 4, 8, dtype=torch.float32)

        send_activation(
            communicator=comm,
            tensor=tensor,
            dst_rank=3,
            request_id="req-001",
            sequence_position=5,
        )

        assert len(comm.sent_tensors) == 1
        sent_tensor, dst = comm.sent_tensors[0]
        assert dst == 3
        assert torch.equal(sent_tensor, tensor)

    def test_returns_activation_message_with_correct_metadata(self) -> None:
        comm = MockCommunicator()
        tensor = torch.randn(1, 16, 64, dtype=torch.float16)

        message = send_activation(
            communicator=comm,
            tensor=tensor,
            dst_rank=1,
            request_id="req-abc",
            sequence_position=42,
        )

        assert isinstance(message, ActivationMessage)
        assert message.request_id == "req-abc"
        assert message.sequence_position == 42
        assert message.tensor_shape == (1, 16, 64)
        assert message.tensor_dtype == "float16"

    def test_supports_bfloat16(self) -> None:
        comm = MockCommunicator()
        tensor = torch.randn(4, 8, dtype=torch.bfloat16)

        message = send_activation(
            communicator=comm,
            tensor=tensor,
            dst_rank=2,
            request_id="req-bf16",
            sequence_position=0,
        )

        assert message.tensor_dtype == "bfloat16"

    def test_supports_float32(self) -> None:
        comm = MockCommunicator()
        tensor = torch.randn(3, 5, dtype=torch.float32)

        message = send_activation(
            communicator=comm,
            tensor=tensor,
            dst_rank=0,
            request_id="req-f32",
            sequence_position=10,
        )

        assert message.tensor_dtype == "float32"

    def test_rejects_unsupported_dtype(self) -> None:
        comm = MockCommunicator()
        tensor = torch.randint(0, 10, (2, 3), dtype=torch.int64)

        with pytest.raises(ValueError, match="Unsupported tensor dtype"):
            send_activation(
                communicator=comm,
                tensor=tensor,
                dst_rank=1,
                request_id="req-bad",
                sequence_position=0,
            )


# ---------------------------------------------------------------------------
# Tests for recv_activation
# ---------------------------------------------------------------------------


class TestRecvActivation:
    """Tests for recv_activation()."""

    def test_receives_tensor_with_correct_shape_and_dtype(self) -> None:
        expected = torch.ones(2, 4, 8, dtype=torch.float16, device="cpu")
        comm = MockCommunicator(recv_return=expected)

        tensor, message = recv_activation(
            communicator=comm,
            src_rank=0,
            shape=(2, 4, 8),
            dtype="float16",
            target_device="cpu",
            request_id="req-recv-1",
            sequence_position=7,
        )

        assert torch.equal(tensor, expected)
        assert message.request_id == "req-recv-1"
        assert message.sequence_position == 7
        assert message.tensor_shape == (2, 4, 8)
        assert message.tensor_dtype == "float16"

    def test_returns_activation_message_metadata(self) -> None:
        comm = MockCommunicator()

        _, message = recv_activation(
            communicator=comm,
            src_rank=2,
            shape=(1, 32, 128),
            dtype="bfloat16",
            target_device="cpu",
            request_id="req-meta",
            sequence_position=99,
        )

        assert isinstance(message, ActivationMessage)
        assert message.request_id == "req-meta"
        assert message.sequence_position == 99
        assert message.tensor_shape == (1, 32, 128)
        assert message.tensor_dtype == "bfloat16"

    def test_rejects_unknown_dtype_string(self) -> None:
        comm = MockCommunicator()

        with pytest.raises(ValueError, match="Unknown dtype string"):
            recv_activation(
                communicator=comm,
                src_rank=1,
                shape=(2, 2),
                dtype="int8",
                target_device="cpu",
                request_id="req-bad",
                sequence_position=0,
            )


# ---------------------------------------------------------------------------
# Tests for ActivationMessage
# ---------------------------------------------------------------------------


class TestActivationMessage:
    """Tests for the ActivationMessage dataclass."""

    def test_is_frozen(self) -> None:
        msg = ActivationMessage(
            request_id="r1",
            sequence_position=0,
            tensor_shape=(1, 2, 3),
            tensor_dtype="float32",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            msg.request_id = "r2"  # type: ignore[misc]

    def test_equality(self) -> None:
        msg1 = ActivationMessage(
            request_id="r1",
            sequence_position=5,
            tensor_shape=(4, 8),
            tensor_dtype="float16",
        )
        msg2 = ActivationMessage(
            request_id="r1",
            sequence_position=5,
            tensor_shape=(4, 8),
            tensor_dtype="float16",
        )
        assert msg1 == msg2
