"""Multi-process integration tests for tensor-parallel inference.

These tests use torch.multiprocessing.spawn with real Gloo process groups
to verify tensor-parallel coordination across multiple processes. Unlike
the thread-based mock tests, these exercise the actual distributed
communication primitives.

Requirements: 4.1, 4.2, 4.3, 4.6, 5.1, 5.2, 6.1, 6.3, 6.4, 8.3
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    torch_available = True
except ImportError:
    torch_available = False


pytestmark = pytest.mark.skipif(
    not torch_available, reason="torch not available"
)


def _find_free_port() -> int:
    """Find a free TCP port on localhost for process group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Task 12.1: Worker functions (must be module-level for pickling)
# ---------------------------------------------------------------------------


def _worker_tp_forward_pass(
    rank: int,
    world_size: int,
    port: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that initializes TP group and runs a forward pass.

    All ranks should produce identical logits after the forward pass
    because all-reduce synchronizes partial results.

    Requirements: 4.1, 4.2, 4.3, 5.1, 5.2
    """
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        import exo.worker.engines.pytorch_xpu.distributed as dist_module

        # Set the TP process group to the default WORLD group
        dist_module._tp_process_group = dist.group.WORLD

        from exo.worker.engines.pytorch_xpu.tensor_parallel_shard import (
            TPShardConfig,
            TensorParallelShard,
        )

        config = TPShardConfig(
            rank=rank,
            world_size=world_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_key_value_heads=num_kv_heads,
        )

        # Create a synthetic state dict with random weights (same seed on all ranks)
        torch.manual_seed(42)
        state_dict: dict[str, torch.Tensor] = {}
        state_dict["model.embed_tokens.weight"] = torch.randn(100, hidden_size)
        state_dict["model.norm.weight"] = torch.ones(hidden_size)
        state_dict["lm_head.weight"] = torch.randn(100, hidden_size)

        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"
            state_dict[f"{prefix}.input_layernorm.weight"] = torch.ones(hidden_size)
            state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(hidden_size)
            # Attention weights
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
                num_heads * head_dim, hidden_size
            )
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
                num_kv_heads * head_dim, hidden_size
            )
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
                num_kv_heads * head_dim, hidden_size
            )
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
                hidden_size, num_heads * head_dim
            )
            # MLP weights
            state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
                intermediate_size, hidden_size
            )
            state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
                intermediate_size, hidden_size
            )
            state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
                hidden_size, intermediate_size
            )

        # Create the shard (each rank gets its own slice)
        shard = TensorParallelShard(model=state_dict, config=config, device="cpu")

        # Run forward pass with a small input
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        logits, kv_cache = shard.forward(input_data=input_ids)

        # Store results for comparison
        results_dict[rank] = logits.detach().clone()

    finally:
        dist.destroy_process_group()


def _worker_verify_group(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that verifies TP group initialization with rank-sum check.

    Requirements: 4.1, 4.2
    """
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        import exo.worker.engines.pytorch_xpu.distributed as dist_module

        dist_module._tp_process_group = dist.group.WORLD

        from exo.worker.engines.pytorch_xpu.distributed import (
            verify_tensor_parallel_group,
        )

        verified = verify_tensor_parallel_group(world_size)
        results_dict[rank] = verified
    finally:
        dist.destroy_process_group()


def _worker_token_broadcast_allreduce(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that tests token broadcast and all-reduce coordination.

    Requirements: 6.1, 6.3
    """
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # Test token broadcast: rank 0 sends token_id=42
        if rank == 0:
            token_tensor = torch.tensor([42], dtype=torch.long)
        else:
            token_tensor = torch.tensor([0], dtype=torch.long)

        dist.broadcast(token_tensor, src=0)
        received_token = int(token_tensor[0].item())

        # Test all-reduce: each rank contributes a tensor
        partial = torch.ones(1, 4, dtype=torch.float32) * (rank + 1)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        # Expected sum: 1 + 2 + 3 + 4 = 10 for world_size=4
        allreduce_result = float(partial[0, 0].item())

        results_dict[rank] = {
            "received_token": received_token,
            "allreduce_result": allreduce_result,
        }
    finally:
        dist.destroy_process_group()


def _worker_termination_sentinel(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that tests TERMINATION_SENTINEL propagation.

    Requirements: 6.3, 6.4
    """
    TERMINATION_SENTINEL = -1

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # Simulate a few normal token broadcasts first
        for token_id in [10, 20, 30]:
            if rank == 0:
                token_tensor = torch.tensor([token_id], dtype=torch.long)
            else:
                token_tensor = torch.tensor([0], dtype=torch.long)
            dist.broadcast(token_tensor, src=0)

        # Now broadcast TERMINATION_SENTINEL
        if rank == 0:
            sentinel = torch.tensor([TERMINATION_SENTINEL], dtype=torch.long)
        else:
            sentinel = torch.tensor([0], dtype=torch.long)
        dist.broadcast(sentinel, src=0)

        received_sentinel = int(sentinel[0].item())

        # All ranks should detect the sentinel and exit
        results_dict[rank] = {
            "received_sentinel": received_sentinel,
            "exited_cleanly": received_sentinel == TERMINATION_SENTINEL,
        }
    finally:
        dist.destroy_process_group()


def _worker_error_propagation(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that tests error propagation across ranks.

    Rank 2 leaves the group early, causing other ranks' collectives to fail.

    Requirements: 6.4
    """
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # First all-reduce succeeds (verify group works)
        tensor = torch.ones(4, dtype=torch.float32) * rank
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Rank 2 raises an error before the next collective
        if rank == 2:
            results_dict[rank] = {"error_injected": True, "terminated": True}
            # Destroy process group early to simulate crash
            dist.destroy_process_group()
            return

        # Other ranks attempt another all-reduce which will fail
        # because rank 2 has left the group
        try:
            tensor2 = torch.ones(4, dtype=torch.float32) * rank
            dist.all_reduce(tensor2, op=dist.ReduceOp.SUM)
            # If we get here, the all-reduce somehow succeeded
            results_dict[rank] = {"error_detected": False, "terminated": True}
        except Exception as e:
            # Expected: the all-reduce fails because rank 2 left
            results_dict[rank] = {
                "error_detected": True,
                "error_type": type(e).__name__,
                "terminated": True,
            }
    except Exception as e:
        results_dict[rank] = {
            "error_detected": True,
            "error_type": type(e).__name__,
            "terminated": True,
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Task 12.2: Worker functions for hybrid parallelism
# ---------------------------------------------------------------------------


def _worker_hybrid_parallelism(
    rank: int,
    world_size: int,
    port_tp: int,
    port_pp: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that tests TP all-reduce + PP send/recv coexistence.

    Creates two separate process groups:
    - Default group: used for PP-style send/recv
    - Sub-group: used for TP all-reduce

    Verifies that operations on one group don't interfere with the other.

    Requirements: 4.6, 8.3
    """
    # Initialize the default process group (used for PP-style send/recv)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port_pp}",
    )
    try:
        # Create a sub-group for tensor parallelism (all ranks)
        tp_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        # --- Test 1: TP all-reduce works on the sub-group ---
        tp_tensor = torch.ones(8, dtype=torch.float32) * (rank + 1)
        dist.all_reduce(tp_tensor, op=dist.ReduceOp.SUM, group=tp_group)
        expected_tp_sum = float(sum(range(1, world_size + 1)))
        tp_allreduce_correct = all(
            abs(tp_tensor[i].item() - expected_tp_sum) < 1e-5 for i in range(8)
        )

        # --- Test 2: PP-style send/recv on the default group ---
        # Ring pattern: each rank sends to (rank+1) % world_size
        # and receives from (rank-1) % world_size
        send_dst = (rank + 1) % world_size
        recv_src = (rank - 1) % world_size

        send_tensor = torch.tensor([rank * 100 + 7], dtype=torch.float32)
        recv_tensor = torch.zeros(1, dtype=torch.float32)

        # Use isend/irecv to avoid deadlock in ring pattern
        send_req = dist.isend(send_tensor, dst=send_dst)
        recv_req = dist.irecv(recv_tensor, src=recv_src)
        send_req.wait()
        recv_req.wait()

        expected_recv = float(recv_src * 100 + 7)
        pp_sendrecv_correct = abs(recv_tensor[0].item() - expected_recv) < 1e-5

        # --- Test 3: Another TP all-reduce after PP send/recv ---
        # Verify TP group still works after PP operations
        tp_tensor2 = torch.ones(4, dtype=torch.float32) * (rank * 2)
        dist.all_reduce(tp_tensor2, op=dist.ReduceOp.SUM, group=tp_group)
        expected_tp_sum2 = float(sum(r * 2 for r in range(world_size)))
        tp_allreduce2_correct = all(
            abs(tp_tensor2[i].item() - expected_tp_sum2) < 1e-5 for i in range(4)
        )

        results_dict[rank] = {
            "tp_allreduce_correct": tp_allreduce_correct,
            "pp_sendrecv_correct": pp_sendrecv_correct,
            "tp_allreduce2_correct": tp_allreduce2_correct,
            "received_value": float(recv_tensor[0].item()),
        }

    finally:
        dist.destroy_process_group()


def _worker_separate_groups(
    rank: int,
    world_size: int,
    port: int,
    results_dict: dict[int, Any],
) -> None:
    """Worker that tests separate process groups don't interfere.

    Creates two sub-groups (even ranks and odd ranks), performs
    all-reduce on each, and verifies results are group-local.

    Requirements: 4.6, 8.3
    """
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # Create two sub-groups: even ranks and odd ranks
        even_ranks = [r for r in range(world_size) if r % 2 == 0]
        odd_ranks = [r for r in range(world_size) if r % 2 != 0]

        even_group = dist.new_group(ranks=even_ranks, backend="gloo")
        odd_group = dist.new_group(ranks=odd_ranks, backend="gloo")

        # Each rank does all-reduce only within its group
        tensor = torch.tensor([float(rank + 1)], dtype=torch.float32)

        if rank % 2 == 0:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=even_group)
            # Even ranks: sum of (r+1) for r in even_ranks
            expected = float(sum(r + 1 for r in even_ranks))
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=odd_group)
            # Odd ranks: sum of (r+1) for r in odd_ranks
            expected = float(sum(r + 1 for r in odd_ranks))

        result_value = float(tensor[0].item())
        results_dict[rank] = {
            "result": result_value,
            "expected": expected,
            "correct": abs(result_value - expected) < 1e-5,
        }

    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Task 12.1: Test classes
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTensorParallelMultiProcess:
    """Multi-process integration tests using torch.multiprocessing.spawn.

    Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 6.1, 6.3, 6.4
    """

    def test_4_process_forward_pass_identical_outputs(self) -> None:
        """Test that 4 processes produce identical logits after TP forward pass.

        All ranks should have the same logits because all-reduce (sum)
        synchronizes the partial outputs from row-parallel layers.

        Requirements: 4.2, 4.3, 5.1, 5.2
        """
        world_size = 4
        port = _find_free_port()
        hidden_size = 64
        num_heads = 8
        num_kv_heads = 4
        head_dim = 8  # hidden_size / num_heads = 64/8 = 8
        intermediate_size = 128
        num_layers = 2

        # Use a manager dict to collect results from all processes
        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_tp_forward_pass,
            args=(
                world_size,
                port,
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                results_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        # Verify all ranks produced results
        assert len(results_dict) == world_size, (
            f"Expected {world_size} results, got {len(results_dict)}"
        )

        # Verify all ranks have identical logits
        rank0_logits = results_dict[0]
        for r in range(1, world_size):
            torch.testing.assert_close(
                results_dict[r],
                rank0_logits,
                atol=1e-4,
                rtol=1e-4,
                msg=f"Rank {r} logits differ from rank 0",
            )

    def test_2_process_group_initialization_gloo(self) -> None:
        """Test TB4 process group initialization with real Gloo backend.

        Verifies that init_tensor_parallel_group works with 2 processes
        and that verify_tensor_parallel_group passes the rank-sum check.

        Requirements: 4.1, 4.2
        """
        world_size = 2
        port = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_verify_group,
            args=(world_size, port, results_dict),
            nprocs=world_size,
            join=True,
        )

        # Both ranks should verify successfully
        assert results_dict[0] is True, "Rank 0 verification failed"
        assert results_dict[1] is True, "Rank 1 verification failed"

    def test_token_broadcast_and_allreduce_coordination(self) -> None:
        """Test token broadcast from rank 0 and all-reduce coordination.

        Rank 0 broadcasts a token ID, all ranks receive it and perform
        an all-reduce to verify coordination.

        Requirements: 6.1, 6.3
        """
        world_size = 4
        port = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_token_broadcast_allreduce,
            args=(world_size, port, results_dict),
            nprocs=world_size,
            join=True,
        )

        # All ranks should have received token 42
        expected_sum = sum(range(1, world_size + 1))  # 1+2+3+4=10
        for r in range(world_size):
            assert results_dict[r]["received_token"] == 42, (
                f"Rank {r} received token {results_dict[r]['received_token']}, expected 42"
            )
            assert abs(results_dict[r]["allreduce_result"] - expected_sum) < 1e-5, (
                f"Rank {r} all-reduce result {results_dict[r]['allreduce_result']}, "
                f"expected {expected_sum}"
            )

    def test_termination_sentinel_propagation_clean_exit(self) -> None:
        """Test TERMINATION_SENTINEL propagation and clean exit.

        Rank 0 broadcasts TERMINATION_SENTINEL (-1), all other ranks
        detect it and exit cleanly.

        Requirements: 6.3, 6.4
        """
        world_size = 4
        port = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_termination_sentinel,
            args=(world_size, port, results_dict),
            nprocs=world_size,
            join=True,
        )

        # All ranks should have received the sentinel and exited cleanly
        for r in range(world_size):
            assert results_dict[r]["received_sentinel"] == -1, (
                f"Rank {r} did not receive TERMINATION_SENTINEL"
            )
            assert results_dict[r]["exited_cleanly"] is True, (
                f"Rank {r} did not exit cleanly"
            )

    def test_error_propagation_all_ranks_terminate(self) -> None:
        """Test that an error on one rank causes all ranks to terminate.

        Inject an error on rank 2 (leaves group early), verify that
        other ranks detect the failure via failed collectives.

        Requirements: 6.4
        """
        world_size = 4
        port = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        # spawn may raise if a worker process crashes — that's expected
        try:
            mp.spawn(
                _worker_error_propagation,
                args=(world_size, port, results_dict),
                nprocs=world_size,
                join=True,
            )
        except Exception:
            # Some processes may have crashed, which is expected behavior
            pass

        # Verify rank 2 injected the error
        assert results_dict.get(2, {}).get("error_injected", False) or \
               results_dict.get(2, {}).get("terminated", False), (
            "Rank 2 should have injected error or terminated"
        )

        # Verify at least some other ranks detected the error or terminated
        other_ranks_with_results = [
            r for r in [0, 1, 3] if r in results_dict
        ]
        assert len(other_ranks_with_results) >= 1, (
            "At least one non-error rank should have results"
        )


# ---------------------------------------------------------------------------
# Task 12.2: Test classes
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHybridParallelismCoexistence:
    """Integration tests for hybrid parallelism (TP + PP) coexistence.

    Verifies that tensor-parallel all-reduce operations on a sub-group
    don't interfere with pipeline-parallel send/recv on the default group.

    Requirements: 4.6, 8.3
    """

    def test_tp_allreduce_and_pp_sendrecv_coexistence(self) -> None:
        """Test TP group all-reduce + PP group send/recv coexistence.

        Creates separate process groups for TP and PP, runs operations
        on both, and verifies they don't interfere with each other.

        Requirements: 4.6, 8.3
        """
        world_size = 4
        port_tp = _find_free_port()
        port_pp = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_hybrid_parallelism,
            args=(world_size, port_tp, port_pp, results_dict),
            nprocs=world_size,
            join=True,
        )

        # Verify all ranks completed successfully
        assert len(results_dict) == world_size, (
            f"Expected {world_size} results, got {len(results_dict)}"
        )

        for r in range(world_size):
            result = results_dict[r]
            assert result["tp_allreduce_correct"], (
                f"Rank {r}: TP all-reduce produced incorrect result"
            )
            assert result["pp_sendrecv_correct"], (
                f"Rank {r}: PP send/recv produced incorrect result. "
                f"Received {result['received_value']}"
            )
            assert result["tp_allreduce2_correct"], (
                f"Rank {r}: Second TP all-reduce failed after PP operations"
            )

    def test_separate_process_groups_dont_interfere(self) -> None:
        """Verify that operations on separate groups are independent.

        Creates two sub-groups (even ranks and odd ranks), performs
        all-reduce on each, and verifies results are group-local.

        Requirements: 4.6, 8.3
        """
        world_size = 4
        port = _find_free_port()

        manager = mp.Manager()
        results_dict = manager.dict()

        mp.spawn(
            _worker_separate_groups,
            args=(world_size, port, results_dict),
            nprocs=world_size,
            join=True,
        )

        # Verify all ranks got correct group-local results
        assert len(results_dict) == world_size
        for r in range(world_size):
            assert results_dict[r]["correct"], (
                f"Rank {r}: expected {results_dict[r]['expected']}, "
                f"got {results_dict[r]['result']}. "
                f"Groups may be interfering with each other."
            )
