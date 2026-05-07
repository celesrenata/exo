"""Unit tests for tensor-parallel placement and runner integration.

Tests:
- Model card `supports_tensor` field validation
- Fallback from TB4 to ethernet on init failure
- Placement rejects incompatible models (non-divisible heads)
- Placement falls back to pipeline parallelism when TP unavailable

Requirements: 7.2, 7.5, 8.5, 9.4
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.engines.pytorch_xpu.tb4_topology import TB4Peer, TB4Topology
from exo.worker.engines.pytorch_xpu.tensor_parallel_instance import (
    TensorParallelInstance,
)
from exo.worker.engines.pytorch_xpu.tensor_parallel_placement import (
    ModelCardInfo,
    PlacementResult,
    place_tensor_parallel,
    validate_heads_divisibility,
    validate_model_supports_tensor,
    validate_symmetric_groups,
    validate_topology_for_tp,
)
from exo.worker.engines.pytorch_xpu.tensor_parallel_runner import (
    ConnectResult,
    connect_tensor_parallel_group,
)


# --- Fixtures ---


def _make_topology(
    topology_type: str = "mesh",
    world_size: int = 4,
    local_interfaces: list[str] | None = None,
    local_ips: list[str] | None = None,
) -> TB4Topology:
    """Create a TB4Topology for testing."""
    if local_interfaces is None:
        local_interfaces = ["thunderbolt0"]
    if local_ips is None:
        local_ips = ["10.4.0.1"]

    all_node_ips: dict[str, list[str]] = {}
    for i in range(world_size):
        all_node_ips[f"gremlin-{i + 1}"] = [f"10.4.0.{i + 1}"]

    peers = [
        TB4Peer(
            node_ip=f"10.4.0.{i + 2}",
            interface_name=local_interfaces[0],
            bandwidth_gbps=40.0,
        )
        for i in range(world_size - 1)
    ]

    return TB4Topology(
        topology_type=topology_type,  # type: ignore[arg-type]
        local_interfaces=local_interfaces,
        local_ips=local_ips,
        peers=peers,
        all_node_ips=all_node_ips,
    )


def _make_instance(
    world_size: int = 4,
    node_ids: list[str] | None = None,
) -> TensorParallelInstance:
    """Create a TensorParallelInstance for testing."""
    if node_ids is None:
        node_ids = [f"gremlin-{i + 1}" for i in range(world_size)]

    sorted_nodes = sorted(node_ids)
    rank_assignments = {node: rank for rank, node in enumerate(sorted_nodes)}
    tb4_interface_by_node = {node: "thunderbolt0" for node in sorted_nodes}

    return TensorParallelInstance(
        instance_id="test-instance-001",
        model_id="Qwen/Qwen3.5-4B",
        tp_world_size=world_size,
        tb4_master_addr="10.4.0.1",
        tb4_master_port=29500,
        rank_assignments=rank_assignments,
        tb4_interface_by_node=tb4_interface_by_node,
    )


# --- Test: Model card supports_tensor field validation ---


class TestModelCardValidation:
    """Test model card `supports_tensor` field validation.

    Requirements: 7.1, 7.5
    """

    def test_model_supports_tensor_true(self) -> None:
        """Model with supports_tensor=True passes validation."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        valid, reason = validate_model_supports_tensor(card)
        assert valid is True
        assert "supports tensor parallelism" in reason

    def test_model_supports_tensor_false(self) -> None:
        """Model with supports_tensor=False fails validation."""
        card = ModelCardInfo(
            model_id="some-model/unsupported",
            supports_tensor=False,
            attention_heads=16,
        )
        valid, reason = validate_model_supports_tensor(card)
        assert valid is False
        assert "does not support" in reason

    def test_placement_rejects_unsupported_model(self) -> None:
        """Placement falls back to pipeline for unsupported models.

        Requirements: 7.5
        """
        card = ModelCardInfo(
            model_id="some-model/no-tp",
            supports_tensor=False,
            attention_heads=32,
        )
        topology = _make_topology()
        result = place_tensor_parallel(
            model_card=card,
            topology=topology,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
        )
        assert result.strategy == "pipeline_parallel"
        assert result.instance is None
        assert "does not support" in result.reason


# --- Test: Placement rejects incompatible models (non-divisible heads) ---


class TestHeadsDivisibility:
    """Test placement rejects models with non-divisible attention heads.

    Requirements: 7.2
    """

    def test_divisible_heads_pass(self) -> None:
        """32 heads / 4 world_size = 8 heads per rank (valid)."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        valid, reason = validate_heads_divisibility(card, world_size=4)
        assert valid is True

    def test_non_divisible_heads_fail(self) -> None:
        """7 heads / 4 world_size is not evenly divisible (invalid)."""
        card = ModelCardInfo(
            model_id="odd-model/7-heads",
            supports_tensor=True,
            attention_heads=7,
        )
        valid, reason = validate_heads_divisibility(card, world_size=4)
        assert valid is False
        assert "not divisible" in reason

    def test_non_divisible_kv_heads_fail(self) -> None:
        """KV heads not divisible by world_size should fail."""
        card = ModelCardInfo(
            model_id="model/bad-kv",
            supports_tensor=True,
            attention_heads=32,
            num_key_value_heads=3,  # 3 % 4 != 0
        )
        valid, reason = validate_heads_divisibility(card, world_size=4)
        assert valid is False
        assert "KV heads" in reason

    def test_non_divisible_intermediate_size_fail(self) -> None:
        """Intermediate size not divisible by world_size should fail."""
        card = ModelCardInfo(
            model_id="model/bad-mlp",
            supports_tensor=True,
            attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=101,  # 101 % 4 != 0
        )
        valid, reason = validate_heads_divisibility(card, world_size=4)
        assert valid is False
        assert "intermediate_size" in reason

    def test_placement_rejects_non_divisible_model(self) -> None:
        """Full placement flow rejects model with non-divisible heads.

        Requirements: 7.2
        """
        card = ModelCardInfo(
            model_id="model/13-heads",
            supports_tensor=True,
            attention_heads=13,
        )
        topology = _make_topology()
        result = place_tensor_parallel(
            model_card=card,
            topology=topology,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
        )
        assert result.strategy == "pipeline_parallel"
        assert "not divisible" in result.reason


# --- Test: Placement falls back to pipeline when TP unavailable ---


class TestTopologyFallback:
    """Test placement falls back to pipeline parallelism when TP unavailable.

    Requirements: 8.5
    """

    def test_topology_none_falls_back(self) -> None:
        """None topology (discovery failed) triggers fallback."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        result = place_tensor_parallel(
            model_card=card,
            topology=None,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
        )
        assert result.strategy == "pipeline_parallel"
        assert "None" in result.reason or "failed" in result.reason

    def test_topology_unavailable_falls_back(self) -> None:
        """Unavailable topology triggers fallback."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        topology = _make_topology(topology_type="unavailable", world_size=1)
        result = place_tensor_parallel(
            model_card=card,
            topology=topology,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
        )
        assert result.strategy == "pipeline_parallel"
        assert "unavailable" in result.reason

    def test_topology_insufficient_nodes_falls_back(self) -> None:
        """Topology with fewer nodes than requested triggers fallback."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        # Only 2 nodes available but 4 requested
        topology = _make_topology(world_size=2)
        result = place_tensor_parallel(
            model_card=card,
            topology=topology,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
            requested_world_size=4,
        )
        assert result.strategy == "pipeline_parallel"
        assert "world_size=4" in result.reason

    def test_successful_placement(self) -> None:
        """All checks pass — tensor-parallel instance is created."""
        card = ModelCardInfo(
            model_id="Qwen/Qwen3.5-4B",
            supports_tensor=True,
            attention_heads=32,
        )
        topology = _make_topology(world_size=4)
        result = place_tensor_parallel(
            model_card=card,
            topology=topology,
            node_ids=["gremlin-1", "gremlin-2", "gremlin-3", "gremlin-4"],
        )
        assert result.strategy == "tensor_parallel"
        assert result.instance is not None
        assert result.instance.tp_world_size == 4
        assert result.instance.model_id == "Qwen/Qwen3.5-4B"
        assert len(result.instance.rank_assignments) == 4


# --- Test: Fallback from TB4 to ethernet on init failure ---


class TestRunnerFallback:
    """Test fallback from TB4 to ethernet on init failure.

    Requirements: 9.4
    """

    @pytest.mark.asyncio
    async def test_tb4_success_no_fallback(self) -> None:
        """Successful TB4 init does not trigger ethernet fallback."""
        instance = _make_instance()
        topology = _make_topology()

        with (
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.discover_tb4_topology",
                new_callable=AsyncMock,
                return_value=topology,
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.select_tb4_interface",
                return_value="thunderbolt0",
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.init_tensor_parallel_group",
            ) as mock_init,
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.verify_tensor_parallel_group",
                return_value=True,
            ),
        ):
            result = await connect_tensor_parallel_group(
                instance=instance,
                local_node_id="gremlin-1",
                ethernet_interface="eth0",
            )

        assert result.status == "connected"
        assert result.transport == "tb4"
        assert result.interface_used == "thunderbolt0"

    @pytest.mark.asyncio
    async def test_tb4_failure_falls_back_to_ethernet(self) -> None:
        """TB4 init failure triggers ethernet fallback.

        Requirements: 9.4
        """
        instance = _make_instance()
        topology = _make_topology()

        call_count = 0

        def mock_init_side_effect(config: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (TB4) fails
                raise RuntimeError("TB4 connection timeout")
            # Second call (ethernet) succeeds

        with (
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.discover_tb4_topology",
                new_callable=AsyncMock,
                return_value=topology,
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.select_tb4_interface",
                return_value="thunderbolt0",
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.init_tensor_parallel_group",
                side_effect=mock_init_side_effect,
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.verify_tensor_parallel_group",
                return_value=True,
            ),
        ):
            result = await connect_tensor_parallel_group(
                instance=instance,
                local_node_id="gremlin-1",
                ethernet_interface="eth0",
            )

        assert result.status == "connected"
        assert result.transport == "ethernet"
        assert result.interface_used == "eth0"

    @pytest.mark.asyncio
    async def test_both_tb4_and_ethernet_fail(self) -> None:
        """Both TB4 and ethernet fail — returns failed status."""
        instance = _make_instance()
        topology = _make_topology()

        with (
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.discover_tb4_topology",
                new_callable=AsyncMock,
                return_value=topology,
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.select_tb4_interface",
                return_value="thunderbolt0",
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.init_tensor_parallel_group",
                side_effect=RuntimeError("Connection failed"),
            ),
        ):
            result = await connect_tensor_parallel_group(
                instance=instance,
                local_node_id="gremlin-1",
                ethernet_interface="eth0",
            )

        assert result.status == "failed"
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_no_ethernet_fallback_available(self) -> None:
        """TB4 fails and no ethernet fallback provided — returns failed."""
        instance = _make_instance()
        topology = _make_topology()

        with (
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.discover_tb4_topology",
                new_callable=AsyncMock,
                return_value=topology,
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.select_tb4_interface",
                return_value="thunderbolt0",
            ),
            patch(
                "exo.worker.engines.pytorch_xpu.tensor_parallel_runner.init_tensor_parallel_group",
                side_effect=RuntimeError("TB4 timeout"),
            ),
        ):
            result = await connect_tensor_parallel_group(
                instance=instance,
                local_node_id="gremlin-1",
                ethernet_interface=None,  # No fallback
            )

        assert result.status == "failed"
        assert "no ethernet fallback" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_node_not_in_rank_assignments(self) -> None:
        """Node not in rank_assignments returns failed immediately."""
        instance = _make_instance()

        result = await connect_tensor_parallel_group(
            instance=instance,
            local_node_id="unknown-node",
            ethernet_interface="eth0",
        )

        assert result.status == "failed"
        assert "not found" in result.error_message.lower()


# --- Test: Symmetric group validation ---


class TestSymmetricGroupValidation:
    """Unit tests for validate_symmetric_groups.

    Requirements: 8.4
    """

    def test_empty_list_is_symmetric(self) -> None:
        """Empty group list is vacuously symmetric."""
        assert validate_symmetric_groups([]) is True

    def test_single_group_is_symmetric(self) -> None:
        """Single group is always symmetric."""
        assert validate_symmetric_groups([4]) is True

    def test_equal_groups_are_symmetric(self) -> None:
        """All groups with same size are symmetric."""
        assert validate_symmetric_groups([4, 4, 4]) is True

    def test_unequal_groups_are_not_symmetric(self) -> None:
        """Groups with different sizes are not symmetric."""
        assert validate_symmetric_groups([4, 4, 2]) is False

    def test_all_ones_are_symmetric(self) -> None:
        """All size-1 groups are symmetric."""
        assert validate_symmetric_groups([1, 1, 1, 1]) is True


# --- Test: TensorParallelInstance dataclass ---


class TestTensorParallelInstance:
    """Test TensorParallelInstance dataclass construction."""

    def test_basic_construction(self) -> None:
        """Instance can be constructed with required fields."""
        instance = TensorParallelInstance(
            instance_id="test-001",
            model_id="Qwen/Qwen3.5-4B",
            tp_world_size=4,
            tb4_master_addr="10.4.0.1",
            tb4_master_port=29500,
            rank_assignments={
                "gremlin-1": 0,
                "gremlin-2": 1,
                "gremlin-3": 2,
                "gremlin-4": 3,
            },
            tb4_interface_by_node={
                "gremlin-1": "thunderbolt0",
                "gremlin-2": "thunderbolt0",
                "gremlin-3": "thunderbolt0",
                "gremlin-4": "thunderbolt0",
            },
        )
        assert instance.tp_world_size == 4
        assert instance.pipeline_group_id is None
        assert instance.pipeline_rank is None
        assert instance.pipeline_world_size is None

    def test_hybrid_mode_fields(self) -> None:
        """Instance supports optional hybrid parallelism fields."""
        instance = TensorParallelInstance(
            instance_id="hybrid-001",
            model_id="Qwen/Qwen3.5-4B",
            tp_world_size=2,
            tb4_master_addr="10.4.0.1",
            tb4_master_port=29500,
            rank_assignments={"gremlin-1": 0, "gremlin-2": 1},
            tb4_interface_by_node={
                "gremlin-1": "thunderbolt0",
                "gremlin-2": "thunderbolt0",
            },
            pipeline_group_id="pipeline-group-1",
            pipeline_rank=0,
            pipeline_world_size=2,
        )
        assert instance.pipeline_group_id == "pipeline-group-1"
        assert instance.pipeline_rank == 0
        assert instance.pipeline_world_size == 2

    def test_instance_is_frozen(self) -> None:
        """Instance is immutable (frozen dataclass)."""
        instance = _make_instance()
        with pytest.raises(Exception):  # FrozenInstanceError
            instance.tp_world_size = 8  # type: ignore[misc]
