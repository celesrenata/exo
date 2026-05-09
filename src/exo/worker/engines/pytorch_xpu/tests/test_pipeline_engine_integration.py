"""Unit tests for engine dispatch and generator wiring (pipeline parallelism).

Tests that PyTorchXPUEngine correctly dispatches to the appropriate generator
based on the model type (PipelineParallelShard vs TensorParallelShard) and rank.

Requirements: 9.3, 9.4, 9.5, 8.3
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(
    not torch_available, reason="torch not available"
)


# ---------------------------------------------------------------------------
# Helpers: mock task, mock shard, mock engine construction
# ---------------------------------------------------------------------------


def _make_mock_task() -> Any:
    """Create a mock TextGeneration task with the fields _build_generator expects."""
    from exo.shared.types.common import CommandId, ModelId
    from exo.shared.types.tasks import TextGeneration
    from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
    from exo.shared.types.worker.instances import InstanceId

    task_params = TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content="Hello world")],
        max_output_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    task = TextGeneration(
        command_id=CommandId(),
        task_params=task_params,
        instance_id=InstanceId(),
    )
    return task


def _make_pipeline_parallel_shard(rank: int = 0, world_size: int = 4) -> Any:
    """Create a real PipelineParallelShard with mock layers for testing dispatch."""
    from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
        PipelineParallelShard,
        PipelineStageConfig,
        compute_layer_assignment,
    )

    n_layers = 32
    hidden_size = 64
    vocab_size = 100

    start_layer, end_layer = compute_layer_assignment(n_layers, world_size, rank)
    num_local_layers = end_layer - start_layer

    config = PipelineStageConfig(
        rank=rank,
        world_size=world_size,
        start_layer=start_layer,
        end_layer=end_layer,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=n_layers,
        device="cpu",
    )

    # Create mock layers that pass through hidden states
    layers = torch.nn.ModuleList()
    for _ in range(num_local_layers):
        layer = MagicMock(spec=torch.nn.Module)
        # Layer returns (hidden_states, None) when called
        layer.side_effect = lambda hs, **kwargs: (hs, None)
        layers.append(layer)

    embed_tokens = torch.nn.Embedding(vocab_size, hidden_size) if rank == 0 else None
    lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False) if rank == world_size - 1 else None
    final_norm = torch.nn.LayerNorm(hidden_size) if rank == world_size - 1 else None

    # Patch torch.compile to avoid compilation in tests
    with patch("torch.compile", side_effect=RuntimeError("skip compile in test")):
        shard = PipelineParallelShard(
            layers=layers,
            config=config,
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            final_norm=final_norm,
        )

    return shard


def _make_engine(model: Any, rank: int = 0, world_size: int = 4) -> Any:
    """Create a PyTorchXPUEngine with mocked channels."""
    from exo.worker.engines.pytorch_xpu.engine import PyTorchXPUEngine

    cancel_receiver = MagicMock()
    cancel_receiver.collect.return_value = []

    event_sender = MagicMock()

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "hello"
    tokenizer.eos_token_id = 2
    tokenizer.all_special_ids = [2]
    tokenizer.apply_chat_template.return_value = "formatted prompt"

    engine = PyTorchXPUEngine(
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        device="cpu",
        cancel_receiver=cancel_receiver,
        event_sender=event_sender,
    )
    return engine


# ---------------------------------------------------------------------------
# Tests: Engine dispatch
# ---------------------------------------------------------------------------


class TestEngineDispatchPipelineParallel:
    """Test that the engine dispatches to the correct generator for PP shards.

    Requirements: 9.3, 9.4, 9.5
    """

    @patch(
        "exo.worker.engines.pytorch_xpu.pipeline_generator.pipeline_parallel_generate"
    )
    def test_engine_dispatches_pipeline_generate_for_pp_shard_rank_0(
        self, mock_pp_generate: MagicMock
    ) -> None:
        """Engine dispatches to pipeline_parallel_generate for PP shard on rank 0.

        Requirements: 9.5
        """
        # Setup: mock generator returns a single response then stops
        mock_gen = MagicMock(spec=Generator)
        mock_pp_generate.return_value = mock_gen

        shard = _make_pipeline_parallel_shard(rank=0, world_size=4)
        engine = _make_engine(model=shard, rank=0, world_size=4)
        task = _make_mock_task()

        # Act: call _build_generator
        gen = engine._build_generator(task)

        # Assert: pipeline_parallel_generate was called
        mock_pp_generate.assert_called_once()
        call_kwargs = mock_pp_generate.call_args
        # Verify key arguments
        assert call_kwargs.kwargs["model"] is shard or call_kwargs[1].get("model") is shard or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] is shard
        ) or call_kwargs.kwargs.get("model") is shard

        # The returned generator is the mock
        assert gen is mock_gen

    @patch(
        "exo.worker.engines.pytorch_xpu.pipeline_generator.pipeline_parallel_worker_loop"
    )
    def test_engine_dispatches_worker_loop_for_pp_shard_non_rank_0(
        self, mock_worker_loop: MagicMock
    ) -> None:
        """Engine dispatches to pipeline_parallel_worker_loop for PP shard on rank != 0.

        Requirements: 9.4
        """
        shard = _make_pipeline_parallel_shard(rank=1, world_size=4)
        engine = _make_engine(model=shard, rank=1, world_size=4)
        task = _make_mock_task()

        # Act: call _build_generator — this returns a generator wrapping the worker loop
        gen = engine._build_generator(task)

        # The generator is the _pipeline_worker_loop_generator wrapper.
        # When we exhaust it, it calls pipeline_parallel_worker_loop.
        # Exhaust the generator to trigger the worker loop call.
        results = list(gen)

        # Assert: worker loop was called
        mock_worker_loop.assert_called_once()
        call_kwargs = mock_worker_loop.call_args
        assert call_kwargs.kwargs.get("model") is shard or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] is shard
        ) or call_kwargs.kwargs.get("model") is shard
        assert call_kwargs.kwargs.get("rank") == 1 or (
            len(call_kwargs.args) > 2 and call_kwargs.args[2] == 1
        )

        # Worker loop generator yields nothing (StopIteration signals finish)
        assert results == []

    @patch(
        "exo.worker.engines.pytorch_xpu.tensor_parallel_generator.tensor_parallel_generate"
    )
    def test_engine_dispatches_tp_generator_for_tp_shard(
        self, mock_tp_generate: MagicMock
    ) -> None:
        """Engine still dispatches to tensor_parallel_generate for TensorParallelShard.

        Requirements: 8.3
        """
        # Create a mock that is NOT a PipelineParallelShard (simulates TP shard)
        mock_tp_shard = MagicMock()
        # Ensure isinstance check for PipelineParallelShard returns False
        mock_tp_shard.__class__ = type("TensorParallelShard", (), {})

        mock_gen = MagicMock(spec=Generator)
        mock_tp_generate.return_value = mock_gen

        engine = _make_engine(model=mock_tp_shard, rank=0, world_size=4)
        task = _make_mock_task()

        # Act
        gen = engine._build_generator(task)

        # Assert: tensor_parallel_generate was called (not pipeline)
        mock_tp_generate.assert_called_once()
        assert gen is mock_gen


# ---------------------------------------------------------------------------
# Tests: reset_state() clears KV cache
# ---------------------------------------------------------------------------


class TestResetStateClearsCache:
    """Test that reset_state() clears KV cache and recurrent state.

    Requirements: 5.3, 6.3
    """

    def test_reset_state_clears_kv_cache(self) -> None:
        """reset_state() clears KV cache entries after forward pass populates them."""
        from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
            PipelineParallelShard,
            PipelineStageConfig,
        )

        hidden_size = 64
        vocab_size = 100
        num_layers = 4

        config = PipelineStageConfig(
            rank=0,
            world_size=1,
            start_layer=0,
            end_layer=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            device="cpu",
        )

        # Create real layers that produce KV cache entries
        class MockLayer(torch.nn.Module):
            def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                batch, seq_len, hs = hidden_states.shape
                # Return hidden_states and a fake KV cache tuple
                key = torch.randn(batch, 4, seq_len, hs // 4)
                value = torch.randn(batch, 4, seq_len, hs // 4)
                return hidden_states, (key, value)

        layers = torch.nn.ModuleList([MockLayer() for _ in range(num_layers)])
        embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        final_norm = torch.nn.LayerNorm(hidden_size)

        with patch("torch.compile", side_effect=RuntimeError("skip compile")):
            shard = PipelineParallelShard(
                layers=layers,
                config=config,
                embed_tokens=embed_tokens,
                lm_head=lm_head,
                final_norm=final_norm,
            )

        # Verify initial state: all cache entries are None
        assert len(shard._kv_cache) == num_layers
        assert all(entry is None for entry in shard._kv_cache)

        # Run a forward pass to populate KV cache
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        _output, _kv = shard.forward(input_data=input_ids)

        # Verify cache is populated (at least some entries are not None)
        assert any(entry is not None for entry in shard._kv_cache)

        # Reset state
        shard.reset_state()

        # Verify all cache entries are cleared
        assert len(shard._kv_cache) == num_layers
        assert all(entry is None for entry in shard._kv_cache), (
            "reset_state() did not clear all KV cache entries"
        )

    def test_reset_state_clears_all_entries(self) -> None:
        """reset_state() resets all cache entries to None with correct count."""
        from exo.worker.engines.pytorch_xpu.pipeline_parallel_shard import (
            PipelineParallelShard,
            PipelineStageConfig,
        )

        hidden_size = 64
        vocab_size = 100
        num_layers = 8

        config = PipelineStageConfig(
            rank=1,
            world_size=4,
            start_layer=8,
            end_layer=16,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=32,
            device="cpu",
        )

        num_local_layers = config.num_local_layers  # 8

        # Create mock layers
        class MockLayer(torch.nn.Module):
            def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                batch, seq_len, hs = hidden_states.shape
                key = torch.randn(batch, 4, seq_len, hs // 4)
                value = torch.randn(batch, 4, seq_len, hs // 4)
                return hidden_states, (key, value)

        layers = torch.nn.ModuleList([MockLayer() for _ in range(num_local_layers)])

        with patch("torch.compile", side_effect=RuntimeError("skip compile")):
            shard = PipelineParallelShard(
                layers=layers,
                config=config,
                embed_tokens=None,  # Not rank 0
                lm_head=None,  # Not last rank
                final_norm=None,
            )

        # Manually populate cache to simulate post-forward state
        for i in range(num_local_layers):
            shard._kv_cache[i] = (
                torch.randn(1, 4, 5, 16),
                torch.randn(1, 4, 5, 16),
            )

        # Verify cache is populated
        assert all(entry is not None for entry in shard._kv_cache)

        # Reset
        shard.reset_state()

        # Verify: correct number of entries, all None
        assert len(shard._kv_cache) == num_local_layers
        assert all(entry is None for entry in shard._kv_cache)


# ---------------------------------------------------------------------------
# Tests: Termination signal (EOS broadcast) exits all ranks
# ---------------------------------------------------------------------------


class TestTerminationSignal:
    """Test that the termination sentinel causes worker loops to exit.

    Requirements: 4.3, 11.3
    """

    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    @patch(
        "exo.worker.engines.pytorch_xpu.pipeline_generator.recv_activation"
    )
    def test_termination_signal_exits_worker_loop(
        self,
        mock_recv_activation: MagicMock,
        mock_send: MagicMock,
        mock_recv: MagicMock,
        mock_broadcast: MagicMock,
    ) -> None:
        """Worker loop exits when it receives TERMINATION_SENTINEL via broadcast.

        Requirements: 9.4, 9.5
        """
        from exo.worker.engines.pytorch_xpu.pipeline_generator import (
            TERMINATION_SENTINEL,
            pipeline_parallel_worker_loop,
        )

        shard = _make_pipeline_parallel_shard(rank=1, world_size=4)

        # Mock recv to provide seq_len metadata
        def mock_recv_side_effect(tensor: torch.Tensor, src: int) -> None:
            tensor[0] = 1  # seq_len = 1

        mock_recv.side_effect = mock_recv_side_effect

        # Mock recv_activation to return a hidden state tensor
        mock_recv_activation.return_value = torch.randn(1, 1, 64)

        # Mock the shard's forward to return a hidden state
        shard.forward = MagicMock(return_value=(torch.randn(1, 1, 64), [None] * 8))

        # Mock broadcast to set the token to TERMINATION_SENTINEL
        def mock_broadcast_side_effect(tensor: torch.Tensor, src: int) -> None:
            tensor[0] = TERMINATION_SENTINEL

        mock_broadcast.side_effect = mock_broadcast_side_effect

        # Act: run the worker loop — it should exit after receiving sentinel
        pipeline_parallel_worker_loop(
            model=shard,
            device="cpu",
            rank=1,
            world_size=4,
        )

        # Assert: the loop ran once (received activation, forwarded, then got sentinel)
        mock_recv_activation.assert_called_once()
        # send was called (rank 1 sends to rank 2)
        mock_send.assert_called()

    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.recv")
    @patch("torch.distributed.send")
    @patch(
        "exo.worker.engines.pytorch_xpu.pipeline_generator.recv_activation"
    )
    def test_eos_token_exits_worker_loop(
        self,
        mock_recv_activation: MagicMock,
        mock_send: MagicMock,
        mock_recv: MagicMock,
        mock_broadcast: MagicMock,
    ) -> None:
        """Worker loop exits when it receives an EOS token via broadcast.

        Requirements: 4.3, 11.3
        """
        from exo.worker.engines.pytorch_xpu.pipeline_generator import (
            pipeline_parallel_worker_loop,
        )

        shard = _make_pipeline_parallel_shard(rank=1, world_size=4)

        # Mock recv for seq_len metadata
        def mock_recv_side_effect(tensor: torch.Tensor, src: int) -> None:
            tensor[0] = 1

        mock_recv.side_effect = mock_recv_side_effect

        # Mock recv_activation
        mock_recv_activation.return_value = torch.randn(1, 1, 64)

        # Mock forward
        shard.forward = MagicMock(return_value=(torch.randn(1, 1, 64), [None] * 8))

        # Mock broadcast to return EOS token (token_id=2)
        eos_token_id = 2

        def mock_broadcast_side_effect(tensor: torch.Tensor, src: int) -> None:
            tensor[0] = eos_token_id

        mock_broadcast.side_effect = mock_broadcast_side_effect

        # Create a tokenizer mock with eos_token_id
        tokenizer = MagicMock()
        tokenizer.eos_token_id = eos_token_id
        tokenizer.all_special_ids = [eos_token_id]
        tokenizer.decode.return_value = ""

        # Act: run the worker loop with tokenizer so it knows about EOS
        pipeline_parallel_worker_loop(
            model=shard,
            device="cpu",
            rank=1,
            world_size=4,
            tokenizer=tokenizer,
        )

        # Assert: loop exited after one iteration (EOS detected)
        mock_recv_activation.assert_called_once()
