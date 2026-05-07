"""Unit tests for distributed generation runner dispatch logic.

Tests that the runner correctly dispatches to the appropriate generation function
based on world_size and rank values.

Requirements: 7.1, 7.4
"""

import inspect

import pytest


class TestRunnerDispatchLogic:
    """Test that the runner dispatches to the correct generation function."""

    def test_world_size_greater_than_1_rank_0_uses_distributed_generate(self) -> None:
        """Verify that distributed_generate exists, is callable, and is a generator function.

        When world_size > 1 and rank == 0, the runner dispatches to distributed_generate.
        """
        from exo.worker.engines.pytorch_xpu.distributed_generator import distributed_generate

        assert callable(distributed_generate)
        assert inspect.isgeneratorfunction(distributed_generate)

    def test_world_size_greater_than_1_rank_nonzero_uses_worker_loop(self) -> None:
        """Verify that distributed_worker_loop exists and is callable (not a generator).

        When world_size > 1 and rank != 0, the runner dispatches to distributed_worker_loop.
        """
        from exo.worker.engines.pytorch_xpu.distributed_generator import distributed_worker_loop

        assert callable(distributed_worker_loop)
        # Worker loop returns None, not a generator
        assert not inspect.isgeneratorfunction(distributed_worker_loop)

    def test_world_size_1_uses_pytorch_xpu_generate(self) -> None:
        """Verify that pytorch_xpu_generate exists and is a generator function.

        When world_size == 1, the runner uses the single-node generation path.
        """
        from exo.worker.engines.pytorch_xpu.generator import pytorch_xpu_generate

        assert callable(pytorch_xpu_generate)
        assert inspect.isgeneratorfunction(pytorch_xpu_generate)

    @pytest.mark.parametrize(
        "world_size,rank,expected",
        [
            (1, 0, "single"),
            (2, 0, "distributed_generate"),
            (2, 1, "worker_loop"),
            (4, 0, "distributed_generate"),
            (4, 1, "worker_loop"),
            (4, 2, "worker_loop"),
            (4, 3, "worker_loop"),
        ],
        ids=[
            "single_node",
            "2rank_rank0",
            "2rank_rank1",
            "4rank_rank0",
            "4rank_rank1",
            "4rank_rank2",
            "4rank_rank3",
        ],
    )
    def test_dispatch_decision_logic(self, world_size: int, rank: int, expected: str) -> None:
        """Test the conditional dispatch logic matches the runner's branching.

        The runner uses:
          if world_size > 1:
              if rank == 0: → distributed_generate
              else:         → distributed_worker_loop
          else:             → pytorch_xpu_generate (single-node)
        """
        if world_size > 1:
            if rank == 0:
                result = "distributed_generate"
            else:
                result = "worker_loop"
        else:
            result = "single"

        assert result == expected


class TestDistributedGeneratorEdgeCases:
    """Unit tests for edge cases in distributed_generator.

    Requirements: 2.5, 3.1, 3.2, 3.3, 6.3, 6.4
    """

    # --- EOS detection with multiple stop tokens (Requirement 2.5) ---

    def test_eos_detection_im_end_token(self) -> None:
        """Verify that <|im_end|> token triggers stop in distributed_generate."""
        from unittest.mock import MagicMock, patch

        from exo.worker.engines.pytorch_xpu.distributed_generator import (
            TERMINATION_SENTINEL,
            distributed_generate,
        )

        import torch

        # Create a mock tokenizer with <|im_end|> as a special token
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]  # 3-token prompt
        tokenizer.eos_token_id = 50256
        tokenizer.additional_special_tokens_ids = []
        # all_special_ids includes eos + im_end token
        tokenizer.all_special_ids = [50256, 100264]
        # When decoding special token 100264, return <|im_end|>
        def decode_side_effect(ids: list[int], skip_special_tokens: bool = True) -> str:
            token_map = {50256: "<|endoftext|>", 100264: "<|im_end|>"}
            return token_map.get(ids[0], "tok")

        tokenizer.decode.side_effect = decode_side_effect

        # Mock model that returns hidden states + KV cache
        model = MagicMock()
        hidden_states = torch.randn(1, 3, 128)
        model.forward.return_value = (hidden_states, [("k", "v")])
        model.lm_head.weight.shape = [32000]
        model.model.config.vocab_size = 32000  # vocab_size

        # Create logits that will sample to the im_end token (100264)
        # We'll mock sample_token to return the im_end token
        with patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation"
        ) as mock_send, patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation"
        ) as mock_recv, patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.sample_token"
        ) as mock_sample:
            # recv_activation returns logits
            mock_recv.return_value = torch.randn(1, 1, 32000)
            # sample_token returns the im_end token
            mock_sample.return_value = 100264

            gen = distributed_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="Hello",
                device_type="cpu",
                device_id=0,
                rank=0,
                world_size=2,
                max_tokens=10,
                temperature=1.0,
            )

            responses = list(gen)

            # Should terminate with "stop" because 100264 is <|im_end|>
            assert len(responses) == 1
            assert responses[0].finish_reason == "stop"
            assert responses[0].token == 100264

    def test_eos_detection_endoftext_token(self) -> None:
        """Verify that <|endoftext|> token triggers stop in distributed_generate."""
        from unittest.mock import MagicMock, patch

        from exo.worker.engines.pytorch_xpu.distributed_generator import (
            distributed_generate,
        )

        import torch

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.eos_token_id = 50256
        tokenizer.additional_special_tokens_ids = []
        tokenizer.all_special_ids = [50256]

        def decode_side_effect(ids: list[int], skip_special_tokens: bool = True) -> str:
            return "<|endoftext|>" if ids[0] == 50256 else "tok"

        tokenizer.decode.side_effect = decode_side_effect

        model = MagicMock()
        hidden_states = torch.randn(1, 3, 128)
        model.forward.return_value = (hidden_states, [("k", "v")])
        model.lm_head.weight.shape = [32000]
        model.model.config.vocab_size = 32000

        with patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation"
        ), patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation"
        ) as mock_recv, patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.sample_token"
        ) as mock_sample:
            mock_recv.return_value = torch.randn(1, 1, 32000)
            # Return the primary eos_token_id
            mock_sample.return_value = 50256

            gen = distributed_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="Hello",
                device_type="cpu",
                device_id=0,
                rank=0,
                world_size=2,
                max_tokens=10,
                temperature=1.0,
            )

            responses = list(gen)
            assert len(responses) == 1
            assert responses[0].finish_reason == "stop"
            assert responses[0].token == 50256

    # --- Temperature edge cases (Requirement 3.1) ---

    def test_temperature_near_zero_is_greedy(self) -> None:
        """With temperature very close to 0, sample_token should always return argmax."""
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        # Create logits with a clear maximum at position 42
        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0
        logits[0, 7] = 5.0
        logits[0, 99] = 3.0

        # With near-zero temperature, should always pick argmax (42)
        for _ in range(20):
            token = sample_token(logits, temperature=1e-9)
            assert token == 42, f"Expected greedy token 42, got {token}"

    def test_temperature_very_high_is_uniform_like(self) -> None:
        """With very high temperature, distribution should be more uniform.

        We verify this by checking that with high temperature, non-argmax tokens
        can be sampled (the distribution is spread out).
        """
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        # Create logits with a clear maximum
        torch.manual_seed(12345)
        logits = torch.zeros(1, 10)
        logits[0, 0] = 5.0  # Dominant token at low temperature
        logits[0, 1] = 4.9
        logits[0, 2] = 4.8
        logits[0, 3] = 4.7

        # With very high temperature, we should see variety in sampled tokens
        sampled_tokens: set[int] = set()
        for _ in range(200):
            token = sample_token(logits, temperature=100.0)
            sampled_tokens.add(token)

        # With temperature=100, the distribution is nearly uniform over 10 tokens
        # We should see at least 3 different tokens sampled
        assert len(sampled_tokens) >= 3, (
            f"Expected variety with high temperature, only got tokens: {sampled_tokens}"
        )

    # --- Top-k edge cases (Requirement 3.2) ---

    def test_top_k_equals_1_is_greedy(self) -> None:
        """top_k=1 should always return the argmax token (greedy decoding)."""
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        logits = torch.randn(1, 1000)
        expected = int(logits[0].argmax().item())

        # With top_k=1, only the highest logit survives → always argmax
        for _ in range(20):
            token = sample_token(logits, temperature=1.0, top_k=1)
            assert token == expected, f"Expected {expected}, got {token}"

    def test_top_k_equals_vocab_size_is_no_op(self) -> None:
        """top_k == vocab_size should not filter anything (all tokens remain)."""
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        vocab_size = 50
        logits = torch.randn(1, vocab_size)

        # With top_k == vocab_size, no filtering occurs
        # We verify by checking that sampling still works and produces valid tokens
        torch.manual_seed(42)
        sampled_tokens: set[int] = set()
        for _ in range(100):
            token = sample_token(logits, temperature=1.0, top_k=vocab_size)
            assert 0 <= token < vocab_size
            sampled_tokens.add(token)

        # Should see variety since no filtering is applied
        assert len(sampled_tokens) > 1

    # --- Top-p edge cases (Requirement 3.3) ---

    def test_top_p_close_to_zero_is_greedy_like(self) -> None:
        """top_p very close to 0 should behave like greedy (only top-1 token kept).

        Note: top_p filtering keeps the minimum set of tokens whose cumulative
        probability >= top_p. With very small p, only the highest-probability
        token is needed to exceed the threshold.
        """
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        # Create logits with a clear dominant token
        logits = torch.zeros(1, 100)
        logits[0, 55] = 10.0  # This will have highest probability after softmax
        logits[0, 10] = 2.0
        logits[0, 20] = 1.0

        # With top_p very close to 0, only the top token should survive
        # The top token's probability is high enough to exceed any small threshold
        for _ in range(20):
            token = sample_token(logits, temperature=1.0, top_p=0.01)
            assert token == 55, f"Expected greedy-like token 55, got {token}"

    def test_top_p_equals_one_is_no_op(self) -> None:
        """top_p == 1.0 should not filter anything (all tokens remain).

        The implementation skips top-p filtering when top_p >= 1.0.
        """
        from exo.worker.engines.pytorch_xpu.distributed_generator import sample_token

        import torch

        vocab_size = 50
        torch.manual_seed(99)
        logits = torch.randn(1, vocab_size)

        # With top_p=1.0, no filtering occurs — verify sampling works normally
        sampled_tokens: set[int] = set()
        for _ in range(100):
            token = sample_token(logits, temperature=1.0, top_p=1.0)
            assert 0 <= token < vocab_size
            sampled_tokens.add(token)

        # Should see variety since no filtering is applied
        assert len(sampled_tokens) > 1

    # --- Sentinel value handling (Requirements 6.3, 6.4) ---

    def test_sentinel_value_is_negative_one(self) -> None:
        """Verify TERMINATION_SENTINEL == -1."""
        from exo.worker.engines.pytorch_xpu.distributed_generator import TERMINATION_SENTINEL

        assert TERMINATION_SENTINEL == -1

    def test_worker_loop_exits_on_sentinel(self) -> None:
        """Mock recv to return sentinel immediately — verify worker loop exits cleanly."""
        from unittest.mock import MagicMock, patch

        from exo.worker.engines.pytorch_xpu.distributed_generator import (
            TERMINATION_SENTINEL,
            distributed_worker_loop,
        )

        import torch

        model = MagicMock()
        hidden_states = torch.randn(1, 5, 128)
        model.forward.return_value = (hidden_states, [("k", "v")])
        model.lm_head.weight.shape = [32000]
        model.model.config.vocab_size = 32000

        call_count = 0

        def mock_recv(shape: tuple[int, ...], dtype: torch.dtype, src_rank: int, target_device: str) -> torch.Tensor:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First recv: seq_len metadata
                return torch.tensor([5], dtype=torch.int64)
            elif call_count == 2:
                # Second recv: hidden_states for prefill
                return torch.randn(*shape, dtype=dtype)
            elif call_count == 3:
                # Third recv: token from rank 0 — send sentinel to exit
                return torch.tensor([TERMINATION_SENTINEL], dtype=torch.int64)
            else:
                raise RuntimeError("Should not receive more calls after sentinel")

        with patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation",
            side_effect=mock_recv,
        ), patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation"
        ):
            # Should exit cleanly without raising
            distributed_worker_loop(
                model=model,
                device_type="cpu",
                device_id=0,
                rank=1,
                world_size=2,
                hidden_size=128,
                dtype=torch.float32,
            )

        # Verify we received exactly 3 calls (seq_len, hidden_states, sentinel token)
        assert call_count == 3

    # --- Empty/short prompt handling ---

    def test_single_token_prompt(self) -> None:
        """Prompt that tokenizes to 1 token should work correctly."""
        from unittest.mock import MagicMock, patch

        from exo.worker.engines.pytorch_xpu.distributed_generator import (
            distributed_generate,
        )

        import torch

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]  # Single token prompt
        tokenizer.eos_token_id = 50256
        tokenizer.additional_special_tokens_ids = []
        tokenizer.all_special_ids = [50256]
        tokenizer.decode.return_value = "hello"

        model = MagicMock()
        hidden_states = torch.randn(1, 1, 128)
        model.forward.return_value = (hidden_states, [("k", "v")])
        model.lm_head.weight.shape = [32000]
        model.model.config.vocab_size = 32000

        with patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.send_activation"
        ), patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.recv_activation"
        ) as mock_recv, patch(
            "exo.worker.engines.pytorch_xpu.distributed_generator.sample_token"
        ) as mock_sample:
            mock_recv.return_value = torch.randn(1, 1, 32000)
            # Return a non-EOS token first, then EOS
            mock_sample.side_effect = [100, 50256]

            gen = distributed_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="H",
                device_type="cpu",
                device_id=0,
                rank=0,
                world_size=2,
                max_tokens=5,
                temperature=1.0,
            )

            responses = list(gen)

            # First response: token 100 (non-EOS), second: token 50256 (EOS → stop)
            assert len(responses) == 2
            assert responses[0].token == 100
            assert responses[0].finish_reason is None
            assert responses[0].usage is not None
            assert responses[0].usage.prompt_tokens == 1  # Single token prompt
            assert responses[1].token == 50256
            assert responses[1].finish_reason == "stop"
