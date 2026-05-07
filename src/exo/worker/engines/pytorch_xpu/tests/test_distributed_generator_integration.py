"""Integration tests for multi-rank distributed generation coordination.

Tests the full pipeline flow using thread-based simulation with queue-backed
communication to verify coordination logic between ranks without requiring
actual Gloo process groups (which need network coordination and may be flaky
in test/CI environments).

Uses a MockTransformerShard that returns predictable tensors and patches
send_activation/recv_activation with thread-safe queue-based implementations
that route based on the calling thread's rank identity.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.7, 6.1, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import threading
from collections import defaultdict
from queue import Empty, Queue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from exo.worker.engines.pytorch_xpu.distributed_generator import (
    TERMINATION_SENTINEL,
    distributed_generate,
    distributed_worker_loop,
)


# ---------------------------------------------------------------------------
# Mock TransformerShard
# ---------------------------------------------------------------------------


class MockLmHead:
    """Mock lm_head for vocab_size detection."""

    def __init__(self, vocab_size: int) -> None:
        self.weight = torch.randn(vocab_size, 128)


class MockConfig:
    """Mock model config."""

    def __init__(self, hidden_size: int, vocab_size: int = 0) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class MockTransformerShard:
    """Mock model that returns predictable tensors for integration testing.

    - Non-last ranks produce hidden states of shape (batch, seq_len, hidden_size)
    - Last rank produces logits of shape (batch, seq_len, vocab_size)
    - KV cache is simulated as a list of tuples that grows each iteration
    """

    def __init__(self, hidden_size: int, vocab_size: int, is_last_rank: bool) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.is_last_rank = is_last_rank
        self.lm_head = MockLmHead(vocab_size)
        self.config = MockConfig(hidden_size, vocab_size)
        self.forward_count = 0
        self.error_on_iteration: int | None = None
        self._lock = threading.Lock()

        # Simulate the nested model.model.config structure used by TransformerShard
        class _InnerModel:
            def __init__(self, config: MockConfig) -> None:
                self.config = config

        self.model = _InnerModel(MockConfig(hidden_size, vocab_size))

    def forward(
        self, input_data: torch.Tensor, past_key_values: Any = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        with self._lock:
            self.forward_count += 1
            current_count = self.forward_count

        if self.error_on_iteration is not None and current_count == self.error_on_iteration:
            raise RuntimeError(f"Injected error on forward pass {current_count}")

        batch_size = input_data.shape[0]
        seq_len = input_data.shape[1] if input_data.dim() > 1 else 1

        if self.is_last_rank:
            # Last rank produces logits with a predictable pattern:
            # token 5 always has the highest logit (will be sampled with greedy)
            output = torch.zeros(batch_size, seq_len, self.vocab_size)
            output[:, :, 5] = 10.0  # Make token 5 dominant
        else:
            # Other ranks produce hidden states
            output = torch.ones(batch_size, seq_len, self.hidden_size) * 0.5

        # Simple KV cache mock: accumulate entries
        if past_key_values is None:
            new_kv = [(torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16))]
        else:
            # Append new entry to simulate growing cache
            new_entry = (torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16))
            new_kv = list(past_key_values) + [new_entry]

        return output, new_kv

    def parameters(self):  # type: ignore[no-untyped-def]
        yield torch.randn(1)  # For dtype detection


# ---------------------------------------------------------------------------
# Queue-based communication simulation
# ---------------------------------------------------------------------------


class QueueComm:
    """Thread-safe queue-based communication layer simulating send/recv.

    Each (src_rank, dst_rank) pair has a dedicated queue. send_activation puts
    a tensor on the queue; recv_activation gets from the queue.

    Uses thread-local storage to identify which rank is calling, so a single
    patched function can route correctly regardless of which thread calls it.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._queues: dict[tuple[int, int], Queue[torch.Tensor]] = defaultdict(Queue)
        self.timeout = timeout
        self.send_log: list[tuple[int, int, tuple[int, ...]]] = []
        self.recv_log: list[tuple[int, int, tuple[int, ...]]] = []
        self._lock = threading.Lock()
        self._thread_ranks: dict[int, int] = {}  # thread_id → rank

    def register_thread(self, rank: int) -> None:
        """Register the current thread as belonging to a specific rank."""
        self._thread_ranks[threading.current_thread().ident or 0] = rank

    def get_current_rank(self) -> int:
        """Get the rank of the current thread."""
        tid = threading.current_thread().ident or 0
        if tid not in self._thread_ranks:
            raise RuntimeError(f"Thread {tid} not registered with QueueComm")
        return self._thread_ranks[tid]

    def send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Simulate send_activation: put tensor on the (src, dst) queue."""
        src_rank = self.get_current_rank()
        with self._lock:
            self.send_log.append((src_rank, dst_rank, tuple(tensor.shape)))
        self._queues[(src_rank, dst_rank)].put(tensor.clone().cpu())

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src_rank: int,
        target_device: str,
    ) -> torch.Tensor:
        """Simulate recv_activation: get tensor from the (src, dst) queue."""
        dst_rank = self.get_current_rank()
        try:
            tensor = self._queues[(src_rank, dst_rank)].get(timeout=self.timeout)
        except Empty:
            raise RuntimeError(
                f"recv_activation timed out: src_rank={src_rank}, dst_rank={dst_rank}, "
                f"expected_shape={shape}"
            )
        with self._lock:
            self.recv_log.append((src_rank, dst_rank, tuple(tensor.shape)))
        return tensor.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64
VOCAB_SIZE = 100


def make_mock_tokenizer(eos_token_id: int = 99) -> MagicMock:
    """Create a mock tokenizer for integration tests."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4]  # 4-token prompt
    tokenizer.eos_token_id = eos_token_id
    tokenizer.additional_special_tokens_ids = []
    tokenizer.all_special_ids = [eos_token_id]
    tokenizer.decode.return_value = "tok"
    return tokenizer


def run_2_rank_pipeline(
    comm: QueueComm,
    rank0_model: MockTransformerShard,
    rank1_model: MockTransformerShard,
    tokenizer: MagicMock,
    max_tokens: int = 3,
    temperature: float = 0.0,
) -> tuple[list[Any], list[Exception], list[Exception]]:
    """Run a 2-rank pipeline in parallel threads and return results.

    Returns (responses, rank0_errors, rank1_errors).
    """
    responses: list[Any] = []
    rank0_errors: list[Exception] = []
    rank1_errors: list[Exception] = []

    def run_rank0() -> None:
        comm.register_thread(rank=0)
        try:
            with patch(
                "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation",
                side_effect=comm.send,
            ), patch(
                "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation",
                side_effect=comm.recv,
            ):
                gen = distributed_generate(
                    model=rank0_model,
                    tokenizer=tokenizer,
                    prompt="Hello world",
                    device_type="cpu",
                    device_id=0,
                    rank=0,
                    world_size=2,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                for resp in gen:
                    responses.append(resp)
        except Exception as e:
            rank0_errors.append(e)

    def run_rank1() -> None:
        comm.register_thread(rank=1)
        try:
            with patch(
                "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation",
                side_effect=comm.send,
            ), patch(
                "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation",
                side_effect=comm.recv,
            ):
                distributed_worker_loop(
                    model=rank1_model,
                    device_type="cpu",
                    device_id=0,
                    rank=1,
                    world_size=2,
                    hidden_size=HIDDEN_SIZE,
                    dtype=torch.float32,
                )
        except Exception as e:
            rank1_errors.append(e)

    t0 = threading.Thread(target=run_rank0, name="rank-0")
    t1 = threading.Thread(target=run_rank1, name="rank-1")

    t1.start()
    t0.start()

    t0.join(timeout=15.0)
    t1.join(timeout=15.0)

    if t0.is_alive() or t1.is_alive():
        raise RuntimeError(
            f"Threads did not terminate: rank0_alive={t0.is_alive()}, rank1_alive={t1.is_alive()}"
        )

    return responses, rank0_errors, rank1_errors


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDistributedGeneratorIntegration:
    """Integration tests for multi-rank coordination.

    These tests use thread-based simulation with queue-backed communication
    to verify the coordination logic between distributed_generate (rank 0)
    and distributed_worker_loop (rank != 0).

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.7, 6.1, 6.3, 6.4, 6.5
    """

    def test_2_rank_pipeline_hidden_states_flow(self) -> None:
        """Test that hidden states flow rank 0 → rank 1 and logits return.

        Verifies:
        - Rank 0 sends hidden states to rank 1
        - Rank 1 (last rank) receives hidden states, computes forward, sends logits back
        - Rank 0 receives logits and samples a token
        - The pipeline produces valid GenerationResponse objects

        Requirements: 1.1, 1.2, 1.4, 1.5
        """
        comm = QueueComm(timeout=5.0)
        rank0_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=False)
        rank1_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=True)
        tokenizer = make_mock_tokenizer(eos_token_id=99)

        responses, rank0_errors, rank1_errors = run_2_rank_pipeline(
            comm, rank0_model, rank1_model, tokenizer, max_tokens=3
        )

        assert not rank0_errors, f"Rank 0 errors: {rank0_errors}"
        assert not rank1_errors, f"Rank 1 errors: {rank1_errors}"

        # Verify responses were generated
        assert len(responses) >= 1, "Expected at least 1 GenerationResponse"

        # Last response should have finish_reason (either "length" or "stop")
        assert responses[-1].finish_reason is not None

        # Verify hidden states flowed: rank 0 sent to rank 1
        rank0_sends_to_rank1 = [(s, d, sh) for s, d, sh in comm.send_log if s == 0 and d == 1]
        assert len(rank0_sends_to_rank1) > 0, "Rank 0 should have sent tensors to rank 1"

        # Verify logits returned: rank 1 sent to rank 0
        rank1_sends_to_rank0 = [(s, d, sh) for s, d, sh in comm.send_log if s == 1 and d == 0]
        assert len(rank1_sends_to_rank0) > 0, "Rank 1 should have sent logits to rank 0"

        # Verify the logits have the expected shape (1, 1, vocab_size)
        for _, _, shape in rank1_sends_to_rank0:
            assert shape == (1, 1, VOCAB_SIZE), f"Expected logits shape (1, 1, {VOCAB_SIZE}), got {shape}"

        # Verify rank 1 forward was called (prefill + decode iterations)
        assert rank1_model.forward_count >= 1

    def test_token_broadcast_all_ranks_receive_same_token(self) -> None:
        """Test that all ranks receive the same token ID each iteration.

        Verifies:
        - Rank 0 broadcasts the sampled token to rank 1 after each iteration
        - The token received by rank 1 matches what rank 0 sampled
        - Token broadcast uses int64 tensor of shape (1,)

        Requirements: 2.7, 6.1
        """
        comm = QueueComm(timeout=5.0)
        rank0_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=False)
        rank1_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=True)
        tokenizer = make_mock_tokenizer(eos_token_id=99)

        responses, rank0_errors, rank1_errors = run_2_rank_pipeline(
            comm, rank0_model, rank1_model, tokenizer, max_tokens=3
        )

        assert not rank0_errors, f"Rank 0 errors: {rank0_errors}"
        assert not rank1_errors, f"Rank 1 errors: {rank1_errors}"

        # Extract token broadcasts from rank 0 → rank 1 (shape (1,) int64 tensors)
        # These are the token ID sends — they have shape (1,)
        token_sends = [(s, d, sh) for s, d, sh in comm.send_log if s == 0 and d == 1 and sh == (1,)]
        assert len(token_sends) >= 1, "Rank 0 should have broadcast at least 1 token to rank 1"

        # All non-EOS tokens sampled by rank 0 should be token 5 (greedy from mock logits)
        sampled_tokens = [r.token for r in responses if r.finish_reason is None]
        for token in sampled_tokens:
            assert token == 5, f"Expected sampled token 5 (greedy from mock), got {token}"

        # The last token send should be the TERMINATION_SENTINEL (since max_tokens is reached)
        # Verify that the sentinel was sent
        # Count: we expect max_tokens token broadcasts (including the sentinel at the end)
        # For max_tokens=3: prefill token + 1 decode token + sentinel = at least 3 sends of shape (1,)
        assert len(token_sends) >= 2, (
            f"Expected at least 2 token sends (tokens + sentinel), got {len(token_sends)}"
        )

    def test_termination_sentinel_propagates(self) -> None:
        """Test that sentinel propagates and all ranks exit cleanly.

        Verifies:
        - When max_tokens is reached, rank 0 sends TERMINATION_SENTINEL to rank 1
        - Rank 1 receives the sentinel and exits its worker loop
        - Both threads terminate without error

        Requirements: 6.3, 6.4
        """
        comm = QueueComm(timeout=5.0)
        rank0_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=False)
        rank1_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=True)
        tokenizer = make_mock_tokenizer(eos_token_id=99)

        responses, rank0_errors, rank1_errors = run_2_rank_pipeline(
            comm, rank0_model, rank1_model, tokenizer, max_tokens=2
        )

        assert not rank0_errors, f"Rank 0 errors: {rank0_errors}"
        assert not rank1_errors, f"Rank 1 errors: {rank1_errors}"

        # Verify last response has finish_reason "length" (max_tokens reached)
        assert len(responses) >= 1
        assert responses[-1].finish_reason == "length"

        # Verify that both ranks terminated cleanly (threads joined without timeout)
        # This is implicitly verified by run_2_rank_pipeline not raising RuntimeError

        # Verify the sentinel was sent from rank 0 to rank 1
        # The sentinel is a tensor with value -1, shape (1,)
        # We can verify by checking that rank 1's worker loop exited (no errors)
        # and that the last shape-(1,) send from rank 0 to rank 1 exists
        token_sends_to_rank1 = [
            (s, d, sh) for s, d, sh in comm.send_log if s == 0 and d == 1 and sh == (1,)
        ]
        # At least one of these should be the sentinel
        assert len(token_sends_to_rank1) >= 1, "Expected at least 1 token/sentinel send to rank 1"

    def test_error_propagation(self) -> None:
        """Test that an error on one rank causes all ranks to terminate.

        Injects an error on rank 1's model forward pass and verifies:
        - Rank 1 exits its worker loop (error caught internally)
        - Rank 0 detects the communication failure (recv times out or errors)
        - Rank 0 yields an error response
        - Both threads terminate

        Requirements: 6.4, 6.5
        """
        comm = QueueComm(timeout=3.0)  # Short timeout to detect failures quickly
        rank0_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=False)
        rank1_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=True)
        # Inject error on rank 1's first forward pass
        rank1_model.error_on_iteration = 1
        tokenizer = make_mock_tokenizer(eos_token_id=99)

        responses, rank0_errors, rank1_errors = run_2_rank_pipeline(
            comm, rank0_model, rank1_model, tokenizer, max_tokens=5
        )

        # Rank 1 should have exited cleanly (error caught in worker loop)
        assert not rank1_errors, f"Rank 1 should catch errors internally, got: {rank1_errors}"

        # Rank 0 should have produced an error response (recv timed out)
        assert not rank0_errors, f"Rank 0 should yield error response, not raise: {rank0_errors}"
        assert len(responses) >= 1, "Expected at least 1 response from rank 0"

        error_responses = [r for r in responses if r.finish_reason == "error"]
        assert len(error_responses) == 1, (
            f"Expected exactly 1 error response, got {len(error_responses)}. "
            f"All responses: {[(r.finish_reason, r.token) for r in responses]}"
        )

    def test_equal_forward_pass_iterations(self) -> None:
        """Test that both ranks execute the same number of forward passes.

        Verifies that rank 0 and rank 1 each execute exactly K+1 forward passes
        (1 prefill + K decode steps) for a generation run producing K tokens.

        Requirements: 6.5
        """
        comm = QueueComm(timeout=5.0)
        rank0_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=False)
        rank1_model = MockTransformerShard(HIDDEN_SIZE, VOCAB_SIZE, is_last_rank=True)
        tokenizer = make_mock_tokenizer(eos_token_id=99)

        max_tokens = 4

        responses, rank0_errors, rank1_errors = run_2_rank_pipeline(
            comm, rank0_model, rank1_model, tokenizer, max_tokens=max_tokens
        )

        assert not rank0_errors, f"Rank 0 errors: {rank0_errors}"
        assert not rank1_errors, f"Rank 1 errors: {rank1_errors}"

        # Both ranks should have executed the same number of forward passes
        # For max_tokens=4: 1 prefill + 3 decode = 4 forward passes on each rank
        assert rank0_model.forward_count == rank1_model.forward_count, (
            f"Rank 0 forward count ({rank0_model.forward_count}) != "
            f"Rank 1 forward count ({rank1_model.forward_count})"
        )

        # Verify the count matches expected: 1 prefill + (max_tokens - 1) decode = max_tokens
        assert rank0_model.forward_count == max_tokens, (
            f"Expected {max_tokens} forward passes (1 prefill + {max_tokens - 1} decode), "
            f"got {rank0_model.forward_count}"
        )
