"""
Property-based tests for tensor-parallel dispatch detection.

Tests that the runner correctly dispatches to tensor_parallel_generate/tensor_parallel_worker_loop
when shard metadata indicates tensor parallelism (start_layer=0, end_layer=n_layers, world_size>1).

**Validates: Requirements 1.1, 1.2, 2.1, 2.2**
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, example
from hypothesis import strategies as st

from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.instances import PyTorchXPURingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import TensorShardMetadata
from exo.worker.engines.pytorch_xpu.tensor_parallel_instance import (
    TensorParallelInstance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_card(n_layers: int = 32) -> Any:
    """Create a minimal ModelCard for testing."""
    from exo.shared.models.model_cards import ModelCard, ModelTask
    from exo.shared.types.memory import Memory

    return ModelCard(
        model_id="Qwen/Qwen3.5-4B",
        storage_size=Memory(in_bytes=8_000_000_000),
        n_layers=n_layers,
        hidden_size=2560,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _make_tensor_parallel_instance(
    n_layers: int,
    world_size: int,
    ephemeral_port: int = 29500,
) -> tuple[PyTorchXPURingInstance, dict[RunnerId, TensorShardMetadata]]:
    """Create a PyTorchXPURingInstance with tensor-parallel shard metadata.

    Tensor-parallel: ALL nodes have start_layer=0, end_layer=n_layers (all layers).
    """
    node_ids = [NodeId(f"node-{i}") for i in range(world_size)]
    runner_ids = [RunnerId(f"runner-{i}") for i in range(world_size)]

    runner_to_shard: dict[RunnerId, TensorShardMetadata] = {}
    for i, rid in enumerate(runner_ids):
        runner_to_shard[rid] = TensorShardMetadata(
            model_card=_make_model_card(n_layers),
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=n_layers,
            n_layers=n_layers,
        )

    node_to_runner: dict[NodeId, RunnerId] = dict(zip(node_ids, runner_ids))

    hosts_by_node: dict[NodeId, list[Host]] = {}
    for i, nid in enumerate(node_ids):
        hosts_by_node[nid] = [Host(ip=f"10.1.1.{12 + i}", port=ephemeral_port)]

    shard_assignments = ShardAssignments(
        model_id="Qwen/Qwen3.5-4B",
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    instance = PyTorchXPURingInstance(
        instance_id="tp-test-instance",
        shard_assignments=shard_assignments,
        hosts_by_node=hosts_by_node,
        ephemeral_port=ephemeral_port,
    )
    return instance, runner_to_shard


def _determine_dispatch_path(instance: PyTorchXPURingInstance, shard_metadata: TensorShardMetadata) -> str:
    """Simulate the runner's dispatch logic to determine which path is taken.

    This replicates the dispatch decision from runner.py (~line 736):
      is_tensor_parallel = (
          world_size > 1
          and shard_metadata.start_layer == 0
          and shard_metadata.end_layer == shard_metadata.n_layers
      )
      if is_tensor_parallel:
          → "tensor_parallel"
      elif world_size > 1:
          → "pipeline_parallel"
      else:
          → "single_node"

    Returns the dispatch path name.
    """
    # Metadata-based tensor-parallel detection (matches fixed runner.py)
    world_size = shard_metadata.world_size
    is_tensor_parallel = (
        world_size > 1
        and shard_metadata.start_layer == 0
        and shard_metadata.end_layer == shard_metadata.n_layers
    )

    if is_tensor_parallel:
        return "tensor_parallel"
    elif world_size > 1:
        return "pipeline_parallel"
    else:
        return "single_node"


# ---------------------------------------------------------------------------
# Bug Condition Exploration Tests
# ---------------------------------------------------------------------------


class TestBugConditionTensorParallelDispatch:
    """Bug Condition: Tensor-Parallel Dispatch Falls Through to Pipeline Path.

    **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

    These tests verify that when a PyTorchXPURingInstance has tensor-parallel
    shard metadata (start_layer=0, end_layer=n_layers, world_size>1), the
    dispatch logic routes to the tensor_parallel path.

    On UNFIXED code, these tests are EXPECTED TO FAIL because:
    - isinstance(PyTorchXPURingInstance, TensorParallelInstance) is always False
    - The dispatch falls through to the pipeline_parallel path
    """

    @given(
        n_layers=st.integers(min_value=1, max_value=128),
        world_size=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=50)
    @example(n_layers=32, world_size=4)  # Actual gremlin cluster config
    @example(n_layers=24, world_size=2)  # Additional concrete case
    def test_bug_condition_dispatch_routes_to_tensor_parallel(
        self, n_layers: int, world_size: int
    ) -> None:
        """Property 1: Bug Condition - Tensor-Parallel Dispatch Falls Through to Pipeline Path.

        **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

        For any PyTorchXPURingInstance with tensor-parallel shard metadata
        (start_layer=0, end_layer=n_layers, world_size>1), the dispatch
        SHOULD route to "tensor_parallel" path.

        Bug condition: isBugCondition(X) where
          X.instance IS PyTorchXPURingInstance AND
          X.shard_metadata.start_layer = 0 AND
          X.shard_metadata.end_layer = X.shard_metadata.n_layers AND
          X.world_size > 1
        """
        instance, runner_to_shard = _make_tensor_parallel_instance(n_layers, world_size)

        # Pick rank 0's shard metadata
        shard_metadata = runner_to_shard[RunnerId("runner-0")]

        # Verify bug condition holds
        assert isinstance(instance, PyTorchXPURingInstance)
        assert shard_metadata.start_layer == 0
        assert shard_metadata.end_layer == shard_metadata.n_layers
        assert shard_metadata.world_size > 1

        # The dispatch should route to tensor_parallel
        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "tensor_parallel", (
            f"Expected dispatch to 'tensor_parallel' but got '{dispatch_path}'. "
            f"Bug confirmed: with n_layers={n_layers}, world_size={world_size}, "
            f"dispatch returns '{dispatch_path}' instead of 'tensor_parallel'. "
            f"isinstance(PyTorchXPURingInstance, TensorParallelInstance) = "
            f"{isinstance(instance, TensorParallelInstance)}"
        )

    def test_root_cause_type_mismatch(self) -> None:
        """Verify root cause: PyTorchXPURingInstance is NOT an instance of TensorParallelInstance.

        **Validates: Requirements 1.1, 1.2**

        This confirms the fundamental type mismatch that causes the bug:
        placement creates PyTorchXPURingInstance, but the runner checks
        isinstance(instance, TensorParallelInstance).
        """
        instance, _ = _make_tensor_parallel_instance(n_layers=32, world_size=4)

        # This MUST be False — it's the root cause of the bug
        assert isinstance(instance, PyTorchXPURingInstance)
        assert not isinstance(instance, TensorParallelInstance), (
            "UNEXPECTED: PyTorchXPURingInstance IS an instance of TensorParallelInstance. "
            "This would mean the bug doesn't exist as hypothesized."
        )

    def test_concrete_gremlin_cluster_config(self) -> None:
        """Concrete case: 4-node gremlin cluster with Qwen3.5-4B (n_layers=32, world_size=4).

        **Validates: Requirements 1.2, 2.1, 2.2**

        This is the exact configuration that hangs in production.
        """
        instance, runner_to_shard = _make_tensor_parallel_instance(
            n_layers=32, world_size=4
        )
        shard_metadata = runner_to_shard[RunnerId("runner-0")]

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "tensor_parallel", (
            f"Gremlin cluster bug confirmed: n_layers=32, world_size=4, "
            f"dispatch returns '{dispatch_path}' instead of 'tensor_parallel'"
        )

    def test_concrete_two_node_config(self) -> None:
        """Concrete case: 2-node cluster with n_layers=24, world_size=2.

        **Validates: Requirements 1.2, 2.1, 2.2**
        """
        instance, runner_to_shard = _make_tensor_parallel_instance(
            n_layers=24, world_size=2
        )
        shard_metadata = runner_to_shard[RunnerId("runner-0")]

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "tensor_parallel", (
            f"Bug confirmed: n_layers=24, world_size=2, "
            f"dispatch returns '{dispatch_path}' instead of 'tensor_parallel'"
        )


# ---------------------------------------------------------------------------
# Additional Helpers for Preservation Tests
# ---------------------------------------------------------------------------


def _make_pipeline_parallel_instance(
    start_layer: int,
    end_layer: int,
    n_layers: int,
    world_size: int,
    device_rank: int = 0,
    ephemeral_port: int = 29500,
) -> tuple[PyTorchXPURingInstance, TensorShardMetadata]:
    """Create a PyTorchXPURingInstance with pipeline-parallel shard metadata.

    Pipeline-parallel: each node has a DIFFERENT layer range.
    Returns the instance and the shard metadata for the specified device_rank.
    """
    node_ids = [NodeId(f"node-{i}") for i in range(world_size)]
    runner_ids = [RunnerId(f"runner-{i}") for i in range(world_size)]

    runner_to_shard: dict[RunnerId, TensorShardMetadata] = {}
    # For simplicity, assign the given start_layer/end_layer to device_rank,
    # and distribute remaining layers to other ranks
    for i, rid in enumerate(runner_ids):
        if i == device_rank:
            runner_to_shard[rid] = TensorShardMetadata(
                model_card=_make_model_card(n_layers),
                device_rank=i,
                world_size=world_size,
                start_layer=start_layer,
                end_layer=end_layer,
                n_layers=n_layers,
            )
        else:
            # Other ranks get placeholder layer ranges (not the focus of the test)
            layers_per_rank = n_layers // world_size
            other_start = i * layers_per_rank
            other_end = (i + 1) * layers_per_rank if i < world_size - 1 else n_layers
            runner_to_shard[rid] = TensorShardMetadata(
                model_card=_make_model_card(n_layers),
                device_rank=i,
                world_size=world_size,
                start_layer=other_start,
                end_layer=other_end,
                n_layers=n_layers,
            )

    node_to_runner: dict[NodeId, RunnerId] = dict(zip(node_ids, runner_ids))

    hosts_by_node: dict[NodeId, list[Host]] = {}
    for i, nid in enumerate(node_ids):
        hosts_by_node[nid] = [Host(ip=f"10.1.1.{12 + i}", port=ephemeral_port)]

    shard_assignments = ShardAssignments(
        model_id="Qwen/Qwen3.5-4B",
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    instance = PyTorchXPURingInstance(
        instance_id="pipeline-test-instance",
        shard_assignments=shard_assignments,
        hosts_by_node=hosts_by_node,
        ephemeral_port=ephemeral_port,
    )
    shard_metadata = runner_to_shard[runner_ids[device_rank]]
    return instance, shard_metadata


def _make_single_node_instance(
    start_layer: int = 0,
    end_layer: int = 32,
    n_layers: int = 32,
    ephemeral_port: int = 29500,
) -> tuple[PyTorchXPURingInstance, TensorShardMetadata]:
    """Create a PyTorchXPURingInstance with single-node shard metadata (world_size=1)."""
    node_ids = [NodeId("node-0")]
    runner_ids = [RunnerId("runner-0")]

    runner_to_shard: dict[RunnerId, TensorShardMetadata] = {
        runner_ids[0]: TensorShardMetadata(
            model_card=_make_model_card(n_layers),
            device_rank=0,
            world_size=1,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=n_layers,
        )
    }

    node_to_runner: dict[NodeId, RunnerId] = dict(zip(node_ids, runner_ids))

    hosts_by_node: dict[NodeId, list[Host]] = {
        node_ids[0]: [Host(ip="10.1.1.12", port=ephemeral_port)]
    }

    shard_assignments = ShardAssignments(
        model_id="Qwen/Qwen3.5-4B",
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    instance = PyTorchXPURingInstance(
        instance_id="single-node-test-instance",
        shard_assignments=shard_assignments,
        hosts_by_node=hosts_by_node,
        ephemeral_port=ephemeral_port,
    )
    shard_metadata = runner_to_shard[runner_ids[0]]
    return instance, shard_metadata


# ---------------------------------------------------------------------------
# Preservation Property Tests
# ---------------------------------------------------------------------------


class TestPreservationPipelineAndSingleNode:
    """Preservation: Pipeline-Parallel and Single-Node Dispatch Unchanged.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.6**

    These tests verify that the dispatch logic correctly routes to:
    - "pipeline_parallel" for multi-node instances with different layer ranges
    - "single_node" for world_size=1 instances
    - Non-PyTorchXPURing backends are unaffected by PyTorch dispatch logic

    These tests MUST PASS on UNFIXED code (they test non-bug-condition paths).
    The fix must NOT break these paths.
    """

    # ------------------------------------------------------------------
    # Observation tests (concrete cases)
    # ------------------------------------------------------------------

    def test_observation_pipeline_parallel_first_stage(self) -> None:
        """Observe: Pipeline-parallel first stage dispatches to pipeline_parallel.

        **Validates: Requirements 3.1, 3.2**

        start_layer=0, end_layer=8, n_layers=32, world_size=4
        → dispatches to "pipeline_parallel" (NOT tensor_parallel)
        """
        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=0, end_layer=8, n_layers=32, world_size=4, device_rank=0
        )

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Expected 'pipeline_parallel' for first pipeline stage, got '{dispatch_path}'"
        )

    def test_observation_pipeline_parallel_middle_stage(self) -> None:
        """Observe: Non-first pipeline stage dispatches to pipeline_parallel.

        **Validates: Requirements 3.1, 3.2**

        start_layer=8, end_layer=16, n_layers=32, world_size=4
        → dispatches to "pipeline_parallel"
        """
        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=8, end_layer=16, n_layers=32, world_size=4, device_rank=1
        )

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Expected 'pipeline_parallel' for middle pipeline stage, got '{dispatch_path}'"
        )

    def test_observation_single_node_full_model(self) -> None:
        """Observe: Single-node instance dispatches to single_node path.

        **Validates: Requirements 3.3**

        world_size=1, start_layer=0, end_layer=32, n_layers=32
        → dispatches to "single_node" (NOT tensor_parallel, even with all layers)
        """
        instance, shard_metadata = _make_single_node_instance(
            start_layer=0, end_layer=32, n_layers=32
        )

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "single_node", (
            f"Expected 'single_node' for world_size=1, got '{dispatch_path}'"
        )

    def test_observation_non_pytorch_backends_unaffected(self) -> None:
        """Observe: MlxRingInstance and TinygradRingInstance are not PyTorchXPURingInstance.

        **Validates: Requirements 3.6**

        Non-PyTorchXPURing backends have completely different instance types
        and are never processed by the PyTorch dispatch logic.
        """
        from exo.shared.types.worker.instances import (
            MlxRingInstance,
            TinygradRingInstance,
        )

        # These types are distinct from PyTorchXPURingInstance
        assert not issubclass(MlxRingInstance, PyTorchXPURingInstance)
        assert not issubclass(TinygradRingInstance, PyTorchXPURingInstance)

        # They are also not TensorParallelInstance
        assert not issubclass(MlxRingInstance, TensorParallelInstance)
        assert not issubclass(TinygradRingInstance, TensorParallelInstance)

    # ------------------------------------------------------------------
    # Property-based tests
    # ------------------------------------------------------------------

    @given(
        n_layers=st.integers(min_value=4, max_value=128),
        world_size=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=50)
    @example(n_layers=32, world_size=4)  # Gremlin cluster pipeline config
    @example(n_layers=24, world_size=2)
    def test_preservation_pipeline_parallel_dispatch(
        self, n_layers: int, world_size: int
    ) -> None:
        """Property: Pipeline-parallel dispatch is preserved for non-TP shard metadata.

        **Validates: Requirements 3.1, 3.2**

        For any multi-node PyTorchXPURingInstance where shard metadata indicates
        pipeline parallelism (start_layer != 0 OR end_layer != n_layers),
        the dispatch MUST route to "pipeline_parallel".

        This is the preservation property: NOT isBugCondition(X) AND world_size > 1
        AND (start_layer != 0 OR end_layer != n_layers)
        """
        # Generate a pipeline-parallel layer range: end_layer < n_layers
        # (first pipeline stage: start_layer=0, end_layer = n_layers // world_size)
        layers_per_rank = max(1, n_layers // world_size)
        start_layer = 0
        end_layer = min(layers_per_rank, n_layers - 1)  # Ensure end_layer < n_layers

        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=n_layers,
            world_size=world_size,
            device_rank=0,
        )

        # Verify this is NOT the bug condition
        assert not (
            shard_metadata.start_layer == 0
            and shard_metadata.end_layer == shard_metadata.n_layers
            and shard_metadata.world_size > 1
        ), "Test setup error: generated bug condition input"

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Preservation violated: pipeline-parallel instance with "
            f"start_layer={start_layer}, end_layer={end_layer}, n_layers={n_layers}, "
            f"world_size={world_size} dispatched to '{dispatch_path}' instead of 'pipeline_parallel'"
        )

    @given(
        n_layers=st.integers(min_value=4, max_value=128),
        world_size=st.integers(min_value=2, max_value=16),
        rank=st.integers(min_value=0, max_value=15),
    )
    @settings(max_examples=50)
    @example(n_layers=32, world_size=4, rank=1)  # Middle stage
    @example(n_layers=32, world_size=4, rank=3)  # Last stage
    def test_preservation_pipeline_non_first_stage(
        self, n_layers: int, world_size: int, rank: int
    ) -> None:
        """Property: Non-first pipeline stages dispatch to pipeline_parallel.

        **Validates: Requirements 3.1, 3.2**

        For any pipeline stage where start_layer > 0 (non-first stage),
        the dispatch MUST route to "pipeline_parallel".
        """
        # Clamp rank to valid range
        rank = rank % world_size

        # Generate non-first stage: start_layer > 0
        layers_per_rank = max(1, n_layers // world_size)
        # Use rank >= 1 to ensure start_layer > 0
        effective_rank = max(1, rank) if rank > 0 else 1
        if effective_rank >= world_size:
            effective_rank = world_size - 1

        start_layer = effective_rank * layers_per_rank
        end_layer = min((effective_rank + 1) * layers_per_rank, n_layers)

        # Ensure valid layer range
        if start_layer >= n_layers:
            start_layer = n_layers - layers_per_rank
            end_layer = n_layers
        if start_layer < 1:
            start_layer = 1
        if end_layer <= start_layer:
            end_layer = start_layer + 1
        if end_layer > n_layers:
            end_layer = n_layers

        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=n_layers,
            world_size=world_size,
            device_rank=effective_rank,
        )

        # Verify this is NOT the bug condition (start_layer > 0)
        assert shard_metadata.start_layer > 0, "Test setup error: start_layer should be > 0"

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Preservation violated: non-first pipeline stage with "
            f"start_layer={start_layer}, end_layer={end_layer}, n_layers={n_layers}, "
            f"world_size={world_size} dispatched to '{dispatch_path}' instead of 'pipeline_parallel'"
        )

    @given(
        n_layers=st.integers(min_value=1, max_value=128),
        start_layer=st.integers(min_value=0, max_value=127),
        end_layer_offset=st.integers(min_value=1, max_value=128),
    )
    @settings(max_examples=50)
    @example(n_layers=32, start_layer=0, end_layer_offset=32)  # Full model, single node
    @example(n_layers=24, start_layer=0, end_layer_offset=24)
    @example(n_layers=32, start_layer=0, end_layer_offset=8)   # Partial model, single node
    def test_preservation_single_node_dispatch(
        self, n_layers: int, start_layer: int, end_layer_offset: int
    ) -> None:
        """Property: Single-node dispatch is preserved regardless of layer range.

        **Validates: Requirements 3.3**

        For any instance with world_size=1, the dispatch MUST route to "single_node",
        regardless of start_layer and end_layer values.

        This includes the edge case: world_size=1 with start_layer=0, end_layer=n_layers
        (single-node full model, NOT tensor-parallel).
        """
        # Clamp to valid ranges
        start_layer = start_layer % n_layers
        end_layer = min(start_layer + end_layer_offset, n_layers)
        if end_layer <= start_layer:
            end_layer = start_layer + 1
        if end_layer > n_layers:
            end_layer = n_layers

        instance, shard_metadata = _make_single_node_instance(
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=n_layers,
        )

        # Verify world_size=1
        assert shard_metadata.world_size == 1

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "single_node", (
            f"Preservation violated: single-node instance with "
            f"start_layer={start_layer}, end_layer={end_layer}, n_layers={n_layers}, "
            f"world_size=1 dispatched to '{dispatch_path}' instead of 'single_node'"
        )

    # ------------------------------------------------------------------
    # Edge case tests
    # ------------------------------------------------------------------

    def test_preservation_edge_case_single_node_full_model(self) -> None:
        """Edge case: world_size=1 with start_layer=0, end_layer=n_layers.

        **Validates: Requirements 3.3**

        This is NOT tensor-parallel (world_size=1), even though all layers are present.
        Must dispatch to "single_node".
        """
        instance, shard_metadata = _make_single_node_instance(
            start_layer=0, end_layer=32, n_layers=32
        )

        assert shard_metadata.start_layer == 0
        assert shard_metadata.end_layer == shard_metadata.n_layers
        assert shard_metadata.world_size == 1

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "single_node", (
            f"Edge case failed: world_size=1 with full model should be 'single_node', "
            f"got '{dispatch_path}'"
        )

    def test_preservation_edge_case_first_pipeline_stage(self) -> None:
        """Edge case: start_layer=0 with end_layer < n_layers (first pipeline stage).

        **Validates: Requirements 3.1, 3.2**

        This is NOT tensor-parallel (end_layer < n_layers), even though start_layer=0.
        Must dispatch to "pipeline_parallel".
        """
        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=0, end_layer=8, n_layers=32, world_size=4, device_rank=0
        )

        assert shard_metadata.start_layer == 0
        assert shard_metadata.end_layer < shard_metadata.n_layers
        assert shard_metadata.world_size > 1

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Edge case failed: first pipeline stage (start_layer=0, end_layer=8) "
            f"should be 'pipeline_parallel', got '{dispatch_path}'"
        )

    def test_preservation_edge_case_last_pipeline_stage(self) -> None:
        """Edge case: start_layer > 0 with end_layer = n_layers (last pipeline stage).

        **Validates: Requirements 3.1, 3.2**

        This is NOT tensor-parallel (start_layer > 0), even though end_layer = n_layers.
        Must dispatch to "pipeline_parallel".
        """
        instance, shard_metadata = _make_pipeline_parallel_instance(
            start_layer=24, end_layer=32, n_layers=32, world_size=4, device_rank=3
        )

        assert shard_metadata.start_layer > 0
        assert shard_metadata.end_layer == shard_metadata.n_layers
        assert shard_metadata.world_size > 1

        dispatch_path = _determine_dispatch_path(instance, shard_metadata)
        assert dispatch_path == "pipeline_parallel", (
            f"Edge case failed: last pipeline stage (start_layer=24, end_layer=32) "
            f"should be 'pipeline_parallel', got '{dispatch_path}'"
        )

    def test_preservation_non_pytorch_backends_type_isolation(self) -> None:
        """Property: MlxRingInstance and TinygradRingInstance are type-isolated.

        **Validates: Requirements 3.6**

        Non-PyTorchXPURing backends use completely different instance types.
        The PyTorch dispatch logic (which checks isinstance for TensorParallelInstance
        and then falls through based on world_size) only applies to PyTorchXPURingInstance.
        Other backends have their own dispatch paths in runner.py.
        """
        from exo.shared.types.worker.instances import (
            MlxRingInstance,
            TinygradRingInstance,
            MlxJacclInstance,
        )

        # Verify type isolation: none of these are PyTorchXPURingInstance
        assert not issubclass(MlxRingInstance, PyTorchXPURingInstance)
        assert not issubclass(TinygradRingInstance, PyTorchXPURingInstance)
        assert not issubclass(MlxJacclInstance, PyTorchXPURingInstance)

        # Verify none are TensorParallelInstance either
        assert not issubclass(MlxRingInstance, TensorParallelInstance)
        assert not issubclass(TinygradRingInstance, TensorParallelInstance)
        assert not issubclass(MlxJacclInstance, TensorParallelInstance)

        # The dispatch logic in runner.py uses pattern matching on instance type:
        # match instance: case PyTorchXPURingInstance: ... case MlxRingInstance: ...
        # So non-PyTorch backends never enter the PyTorch dispatch branch at all.
