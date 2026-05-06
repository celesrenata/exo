"""Unit tests for tensor-parallel generation pipeline.

Tests the tensor_parallel_generate() orchestrator, tensor_parallel_worker_loop(),
TPPerformanceMetrics, and error handling behavior.

Requirements: 6.3, 6.4, 6.6, 10.1, 10.3, 11.1, 11.2
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestModuleImportsAndConstants:
    """Test that the module exports the expected symbols."""

    def test_termination_sentinel_value(self) -> None:
        """TERMINATION_SENTINEL should be -1."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
        )

        assert TERMINATION_SENTINEL == -1

    def test_tensor_parallel_generate_is_generator_function(self) -> None:
        """tensor_parallel_generate should be a generator function."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_generate,
        )

        assert callable(tensor_parallel_generate)
        assert inspect.isgeneratorfunction(tensor_parallel_generate)

    def test_tensor_parallel_worker_loop_is_not_generator(self) -> None:
        """tensor_parallel_worker_loop should be a regular function (not generator)."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_worker_loop,
        )

        assert callable(tensor_parallel_worker_loop)
        assert not inspect.isgeneratorfunction(tensor_parallel_worker_loop)

    def test_tp_performance_metrics_dataclass(self) -> None:
        """TPPerformanceMetrics should be importable and constructible."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics()
        assert metrics.allreduce_latencies_ms == []
        assert metrics.prefill_time_seconds == 0.0
        assert metrics.decode_tokens_per_second == 0.0
        assert metrics.total_allreduce_bytes == 0
        assert metrics.total_allreduce_time_seconds == 0.0


class TestTPPerformanceMetrics:
    """Test TPPerformanceMetrics calculations.

    Requirements: 10.1, 10.3
    """

    def test_mean_allreduce_empty(self) -> None:
        """Mean should be 0.0 when no latencies recorded."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics()
        assert metrics.mean_allreduce_ms == 0.0

    def test_mean_allreduce_single_value(self) -> None:
        """Mean of a single value should be that value."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics(allreduce_latencies_ms=[5.0])
        assert metrics.mean_allreduce_ms == 5.0

    def test_mean_allreduce_multiple_values(self) -> None:
        """Mean should be the arithmetic average."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics(allreduce_latencies_ms=[2.0, 4.0, 6.0])
        assert metrics.mean_allreduce_ms == pytest.approx(4.0)

    def test_p99_allreduce_empty(self) -> None:
        """P99 should be 0.0 when no latencies recorded."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics()
        assert metrics.p99_allreduce_ms == 0.0

    def test_p99_allreduce_single_value(self) -> None:
        """P99 of a single value should be that value."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics(allreduce_latencies_ms=[7.5])
        assert metrics.p99_allreduce_ms == 7.5

    def test_p99_allreduce_returns_high_percentile(self) -> None:
        """P99 should return a value near the maximum for sorted data."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        # 100 values from 1.0 to 100.0
        latencies = [float(i) for i in range(1, 101)]
        metrics = TPPerformanceMetrics(allreduce_latencies_ms=latencies)
        # p99 index = int(100 * 0.99) = 99, value = 100.0
        assert metrics.p99_allreduce_ms == 100.0

    def test_bandwidth_utilization_zero_time(self) -> None:
        """Bandwidth utilization should be 0.0 when no time spent."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics(
            total_allreduce_bytes=1000,
            total_allreduce_time_seconds=0.0,
        )
        assert metrics.tb4_bandwidth_utilization == 0.0

    def test_bandwidth_utilization_formula(self) -> None:
        """Bandwidth utilization = bytes / time / 5_000_000_000.

        If we transfer 5 GB in 1 second, utilization should be 1.0 (100% of 40 Gbps).
        """
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TB4_BANDWIDTH_BYTES_PER_SECOND,
            TPPerformanceMetrics,
        )

        # Transfer exactly the theoretical bandwidth in 1 second
        metrics = TPPerformanceMetrics(
            total_allreduce_bytes=int(TB4_BANDWIDTH_BYTES_PER_SECOND),
            total_allreduce_time_seconds=1.0,
        )
        assert metrics.tb4_bandwidth_utilization == pytest.approx(1.0)

    def test_bandwidth_utilization_half(self) -> None:
        """Half the theoretical bandwidth should give 0.5 utilization."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TB4_BANDWIDTH_BYTES_PER_SECOND,
            TPPerformanceMetrics,
        )

        metrics = TPPerformanceMetrics(
            total_allreduce_bytes=int(TB4_BANDWIDTH_BYTES_PER_SECOND / 2),
            total_allreduce_time_seconds=1.0,
        )
        assert metrics.tb4_bandwidth_utilization == pytest.approx(0.5)


class TestTerminationSentinelBroadcast:
    """Test that TERMINATION_SENTINEL is broadcast on EOS and max_tokens.

    Requirements: 6.6
    """

    def test_sentinel_broadcast_on_eos_first_token(self) -> None:
        """When first token is EOS, TERMINATION_SENTINEL should be broadcast."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
            tensor_parallel_generate,
        )

        # Mock model that returns logits where token 2 (EOS) has highest prob
        mock_model = MagicMock()
        vocab_size = 10
        logits = torch.full((1, 5, vocab_size), -100.0)
        logits[0, -1, 2] = 100.0  # Token 2 will be sampled (greedy with low temp)
        mock_model.forward.return_value = (logits, [])

        # Mock tokenizer where eos_token_id = 2
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 3, 5, 7, 9]
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.all_special_ids = [2]
        mock_tokenizer.decode.return_value = ""

        broadcast_calls: list[torch.Tensor] = []

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_calls.append(tensor.clone())

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            gen = tensor_parallel_generate(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="hello",
                device="cpu",
                rank=0,
                world_size=4,
                max_tokens=10,
                temperature=0.01,  # Near-greedy
            )
            responses = list(gen)

        # Should have yielded one response with finish_reason="stop"
        assert len(responses) == 1
        assert responses[0].finish_reason == "stop"

        # The last broadcast should be the TERMINATION_SENTINEL
        # Broadcasts: token_count, input_tensor, first_token, sentinel
        assert any(
            int(t[0].item()) == TERMINATION_SENTINEL
            for t in broadcast_calls
            if t.shape == torch.Size([1]) and t.dtype == torch.long
        )

    def test_sentinel_broadcast_on_max_tokens(self) -> None:
        """When max_tokens is reached, TERMINATION_SENTINEL should be broadcast."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
            tensor_parallel_generate,
        )

        # Mock model that always returns token 5 (not EOS)
        mock_model = MagicMock()
        vocab_size = 10
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 5] = 100.0  # Token 5 will always be sampled
        mock_model.forward.return_value = (logits, [])

        # Mock tokenizer with no EOS matching token 5
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.eos_token_id = 99  # Won't match token 5
        mock_tokenizer.all_special_ids = [99]
        mock_tokenizer.decode.return_value = "x"

        broadcast_calls: list[torch.Tensor] = []

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_calls.append(tensor.clone())

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            gen = tensor_parallel_generate(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="hello",
                device="cpu",
                rank=0,
                world_size=4,
                max_tokens=3,
                temperature=0.01,
            )
            responses = list(gen)

        # Should have yielded 3 responses, last with finish_reason="length"
        assert len(responses) == 3
        assert responses[-1].finish_reason == "length"

        # The last broadcast should be TERMINATION_SENTINEL
        sentinel_broadcasts = [
            t for t in broadcast_calls
            if t.shape == torch.Size([1]) and t.dtype == torch.long and int(t[0].item()) == TERMINATION_SENTINEL
        ]
        assert len(sentinel_broadcasts) >= 1


class TestTokenBroadcast:
    """Test that tokens are broadcast from rank 0 to all other ranks.

    Requirements: 6.3
    """

    def test_token_broadcast_uses_dist_broadcast_src_0(self) -> None:
        """All token broadcasts should use src=0."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_generate,
        )

        mock_model = MagicMock()
        vocab_size = 10
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 5] = 100.0
        mock_model.forward.return_value = (logits, [])

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer.all_special_ids = [99]
        mock_tokenizer.decode.return_value = "x"

        broadcast_src_values: list[int] = []

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_src_values.append(src)

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            gen = tensor_parallel_generate(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="hi",
                device="cpu",
                rank=0,
                world_size=4,
                max_tokens=2,
                temperature=0.01,
            )
            list(gen)

        # All broadcasts should have src=0
        assert all(src == 0 for src in broadcast_src_values)
        # Should have multiple broadcasts (token_count, input, tokens, sentinel)
        assert len(broadcast_src_values) >= 4


class TestErrorHandling:
    """Test error handling on all-reduce timeout / communication failure.

    Requirements: 11.1, 11.2
    """

    def test_error_during_forward_yields_error_response(self) -> None:
        """If model.forward() raises RuntimeError, yield error response."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_generate,
        )

        mock_model = MagicMock()
        mock_model.forward.side_effect = RuntimeError(
            "Tensor-parallel all-reduce failed: layer_index=3, "
            "tensor_shape=(1, 1, 2560), timeout=30s, rank=0"
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer.all_special_ids = [99]
        mock_tokenizer.decode.return_value = ""

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            pass

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            gen = tensor_parallel_generate(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="hello",
                device="cpu",
                rank=0,
                world_size=4,
                max_tokens=10,
                temperature=1.0,
            )
            responses = list(gen)

        # Should yield exactly one error response
        assert len(responses) == 1
        assert responses[0].finish_reason == "error"
        assert "all-reduce failed" in responses[0].text

    def test_error_broadcasts_sentinel_best_effort(self) -> None:
        """On error, rank 0 should attempt to broadcast TERMINATION_SENTINEL."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
            tensor_parallel_generate,
        )

        mock_model = MagicMock()

        # First broadcast calls succeed (token_count, input_tensor),
        # then forward raises
        call_count = [0]

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            call_count[0] += 1
            # Let the first two broadcasts succeed (token_count + input_tensor)
            # The forward will fail, then error handler broadcasts sentinel

        mock_model.forward.side_effect = RuntimeError("all-reduce timeout")

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer.all_special_ids = [99]
        mock_tokenizer.decode.return_value = ""

        broadcast_tensors: list[torch.Tensor] = []

        def track_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_tensors.append(tensor.clone())

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=track_broadcast):
            gen = tensor_parallel_generate(
                model=mock_model,
                tokenizer=mock_tokenizer,
                prompt="hi",
                device="cpu",
                rank=0,
                world_size=4,
                max_tokens=5,
                temperature=1.0,
            )
            list(gen)

        # The last broadcast should be the sentinel (best-effort)
        sentinel_found = any(
            t.shape == torch.Size([1]) and t.dtype == torch.long and int(t[0].item()) == TERMINATION_SENTINEL
            for t in broadcast_tensors
        )
        assert sentinel_found

    def test_worker_loop_exits_on_forward_error(self) -> None:
        """Worker loop should exit cleanly if forward() raises."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            tensor_parallel_worker_loop,
        )

        mock_model = MagicMock()
        mock_model.forward.side_effect = RuntimeError("all-reduce timeout")

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            # Simulate receiving token count = 3
            if tensor.shape == torch.Size([1]) and int(tensor[0].item()) == 0:
                tensor[0] = 3

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            # Should not raise — exits cleanly
            tensor_parallel_worker_loop(
                model=mock_model,
                device="cpu",
                rank=1,
                world_size=4,
            )


class TestWorkerLoopTermination:
    """Test that worker loop exits on TERMINATION_SENTINEL.

    Requirements: 6.4, 6.6
    """

    def test_worker_exits_on_sentinel_after_prefill(self) -> None:
        """Worker should exit when sentinel is received as first token after prefill."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
            tensor_parallel_worker_loop,
        )

        mock_model = MagicMock()
        mock_model.forward.return_value = (torch.zeros(1, 3, 10), [])

        broadcast_call_count = [0]

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_call_count[0] += 1
            if broadcast_call_count[0] == 1:
                # First broadcast: token count
                tensor[0] = 3
            elif broadcast_call_count[0] == 2:
                # Second broadcast: input_tensor (fill with dummy tokens)
                tensor[0, 0] = 1
                tensor[0, 1] = 2
                tensor[0, 2] = 3
            elif broadcast_call_count[0] == 3:
                # Third broadcast: first token → send SENTINEL
                tensor[0] = TERMINATION_SENTINEL

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            # Should exit without error
            tensor_parallel_worker_loop(
                model=mock_model,
                device="cpu",
                rank=2,
                world_size=4,
            )

        # Model should have been called once for prefill
        assert mock_model.forward.call_count == 1

    def test_worker_exits_on_sentinel_during_decode(self) -> None:
        """Worker should exit when sentinel is received during decode loop."""
        from exo.worker.engines.pytorch_xpu.tensor_parallel_generator import (
            TERMINATION_SENTINEL,
            tensor_parallel_worker_loop,
        )

        mock_model = MagicMock()
        mock_model.forward.return_value = (torch.zeros(1, 1, 10), [])

        broadcast_call_count = [0]

        def mock_broadcast(tensor: torch.Tensor, src: int = 0, **kwargs: object) -> None:
            broadcast_call_count[0] += 1
            if broadcast_call_count[0] == 1:
                # Token count
                tensor[0] = 2
            elif broadcast_call_count[0] == 2:
                # Input tensor
                tensor[0, 0] = 1
                tensor[0, 1] = 2
            elif broadcast_call_count[0] == 3:
                # First token (not sentinel)
                tensor[0] = 42
            elif broadcast_call_count[0] == 4:
                # Second token → SENTINEL (after one decode step)
                tensor[0] = TERMINATION_SENTINEL

        with patch("exo.worker.engines.pytorch_xpu.tensor_parallel_generator.dist.broadcast", side_effect=mock_broadcast):
            tensor_parallel_worker_loop(
                model=mock_model,
                device="cpu",
                rank=1,
                world_size=4,
            )

        # Model called: 1 prefill + 1 decode step = 2
        assert mock_model.forward.call_count == 2
