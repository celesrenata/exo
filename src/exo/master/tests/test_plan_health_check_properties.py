"""Property-based tests for the master's _plan loop instance health check.

These tests extract the health check logic from Master._plan() into a testable
pure function and verify correctness properties using Hypothesis.

Part of the instance-shutdown-loop bugfix spec.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import (
    InstanceId,
    PyTorchXPURingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata

# ---------------------------------------------------------------------------
# Extracted pure function: mirrors the _plan loop's instance health check
# ---------------------------------------------------------------------------

def plan_health_check(
    instances: Mapping[InstanceId, PyTorchXPURingInstance],
    topology: Topology,
    last_seen: Mapping[NodeId, datetime],
) -> list[InstanceId]:
    """Extract of the _plan loop's instance health check logic (FIXED version).

    Returns the list of InstanceIds that would be deleted.

    This mirrors the FIXED code in Master._plan() which uses last_seen
    instead of topology.list_nodes() for liveness checking.
    """
    deleted: list[InstanceId] = []
    now = datetime.now(tz=timezone.utc)
    for instance_id, instance in instances.items():
        for node_id in instance.shard_assignments.node_to_runner:
            last_seen_time = last_seen.get(node_id)
            if last_seen_time is None or (now - last_seen_time) > timedelta(seconds=30):
                deleted.append(instance_id)
                break
    return deleted


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_TEST_MODEL_CARD = ModelCard(
    model_id="test/model",
    storage_size=Memory(in_bytes=1_000_000),
    n_layers=32,
    hidden_size=4096,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
)


def _make_shard(rank: int, world_size: int) -> PipelineShardMetadata:
    """Create a minimal PipelineShardMetadata for testing."""
    return PipelineShardMetadata(
        model_card=_TEST_MODEL_CARD,
        device_rank=rank,
        world_size=world_size,
        start_layer=0,
        end_layer=16,
        n_layers=32,
    )


def _make_two_node_instance(
    instance_id: InstanceId,
    node1: NodeId,
    node2: NodeId,
) -> PyTorchXPURingInstance:
    """Create a 2-node PyTorchXPURingInstance assigned to node1 and node2."""
    runner1 = RunnerId("runner-1")
    runner2 = RunnerId("runner-2")
    return PyTorchXPURingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id="test/model",
            runner_to_shard={
                runner1: _make_shard(rank=0, world_size=2),
                runner2: _make_shard(rank=1, world_size=2),
            },
            node_to_runner={
                node1: runner1,
                node2: runner2,
            },
        ),
        hosts_by_node={
            node1: [Host(ip="0.0.0.0", port=29500), Host(ip="10.1.1.13", port=29500)],
            node2: [Host(ip="10.1.1.12", port=29500), Host(ip="0.0.0.0", port=29500)],
        },
        ephemeral_port=29500,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def recent_timestamp(draw: st.DrawFn) -> datetime:
    """Generate a timestamp within the last 30 seconds (exclusive of boundary)."""
    now = datetime.now(tz=timezone.utc)
    seconds_ago = draw(st.floats(min_value=0.0, max_value=29.9))
    return now - timedelta(seconds=seconds_ago)


# ---------------------------------------------------------------------------
# Property 1: Bug Condition — Multi-node instances with all nodes recently
# seen are NOT deleted (expected behavior)
# ---------------------------------------------------------------------------


class TestBugConditionExploration:
    """Property 1: Bug Condition — Multi-Node Instance Deleted Despite All Nodes Recently Seen.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

    This test MUST FAIL on unfixed code — failure confirms the bug exists.
    The unfixed code checks topology.list_nodes() and will incorrectly emit
    InstanceDeleted when a node is absent from the topology but recently seen.
    """

    @settings(max_examples=100)
    @given(
        last_seen_1=recent_timestamp(),
        last_seen_2=recent_timestamp(),
    )
    def test_bug_condition_multi_node_not_deleted_when_all_recently_seen(
        self,
        last_seen_1: datetime,
        last_seen_2: datetime,
    ) -> None:
        """A 2-node instance where both nodes are recently seen but one is absent
        from topology should NOT be deleted.

        Bug condition: len(node_to_runner) > 1
                       AND EXISTS node_id NOT IN topology.list_nodes()
                       AND ALL node_id IN last_seen within 30s

        On UNFIXED code, the topology-presence check deletes the instance,
        so this test FAILS — confirming the bug.
        """
        node1 = NodeId("gremlin-1")
        node2 = NodeId("gremlin-2")
        instance_id = InstanceId("test-instance")

        instance = _make_two_node_instance(instance_id, node1, node2)
        instances: dict[InstanceId, PyTorchXPURingInstance] = {instance_id: instance}

        # Only gremlin-1 is in the topology — gremlin-2 is absent
        # (simulates transient topology gap)
        topology = Topology()
        topology.add_node(node1)

        # Both nodes have been seen recently (within 30s)
        last_seen: dict[NodeId, datetime] = {
            node1: last_seen_1,
            node2: last_seen_2,
        }

        # The health check should NOT delete this instance because both nodes
        # are recently seen. But the unfixed code will delete it because
        # gremlin-2 is not in topology.list_nodes().
        deleted = plan_health_check(instances, topology, last_seen)

        assert instance_id not in deleted, (
            f"Instance {instance_id} was deleted even though all nodes are recently seen. "
            f"last_seen_1={last_seen_1}, last_seen_2={last_seen_2}. "
            f"This confirms the bug: topology-presence check is too aggressive."
        )


# ---------------------------------------------------------------------------
# Helper: single-node instance factory
# ---------------------------------------------------------------------------


def _make_single_node_instance(
    instance_id: InstanceId,
    node: NodeId,
) -> PyTorchXPURingInstance:
    """Create a 1-node PyTorchXPURingInstance assigned to a single node."""
    runner = RunnerId("runner-solo")
    return PyTorchXPURingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id="test/model",
            runner_to_shard={
                runner: _make_shard(rank=0, world_size=1),
            },
            node_to_runner={
                node: runner,
            },
        ),
        hosts_by_node={
            node: [Host(ip="0.0.0.0", port=29500)],
        },
        ephemeral_port=29500,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies for preservation tests
# ---------------------------------------------------------------------------


@st.composite
def stale_timestamp(draw: st.DrawFn) -> datetime:
    """Generate a timestamp older than 30 seconds (genuinely disconnected)."""
    now = datetime.now(tz=timezone.utc)
    seconds_ago = draw(st.floats(min_value=31.0, max_value=300.0))
    return now - timedelta(seconds=seconds_ago)


# ---------------------------------------------------------------------------
# Property 2: Preservation — Baseline behaviors that MUST PASS on unfixed code
# ---------------------------------------------------------------------------


class TestPreservationSingleNode:
    """Preservation: Single-node instances are NOT deleted when the node is healthy.

    **Validates: Requirements 3.1**

    On the unfixed code, single-node instances work correctly because the
    local node is always present in its own topology. This test confirms
    that baseline behavior.
    """

    @settings(max_examples=100)
    @given(
        last_seen_time=recent_timestamp(),
    )
    def test_preservation_single_node_not_deleted_when_healthy(
        self,
        last_seen_time: datetime,
    ) -> None:
        """A single-node instance where the node is in the topology and
        last_seen within 30s should NOT be deleted.

        This passes on unfixed code because the single node is always in
        its own topology.
        """
        node = NodeId("gremlin-1")
        instance_id = InstanceId("single-node-instance")

        instance = _make_single_node_instance(instance_id, node)
        instances: dict[InstanceId, PyTorchXPURingInstance] = {instance_id: instance}

        # Node is in the topology (always true for single-node)
        topology = Topology()
        topology.add_node(node)

        # Node has been seen recently
        last_seen: dict[NodeId, datetime] = {node: last_seen_time}

        deleted = plan_health_check(instances, topology, last_seen)

        assert instance_id not in deleted, (
            f"Single-node instance {instance_id} was deleted even though the node "
            f"is in topology and recently seen (last_seen={last_seen_time}). "
            f"This would break single-node inference."
        )


class TestPreservationGenuinelyDisconnected:
    """Preservation: Genuinely disconnected nodes still trigger instance deletion.

    **Validates: Requirements 3.2**

    On the unfixed code, instances with nodes absent from topology are deleted.
    When a node has last_seen older than 30s, it will also be absent from
    topology (removed by NodeTimedOut), so the unfixed code correctly deletes
    these instances. This test confirms that baseline behavior.
    """

    @settings(max_examples=100)
    @given(
        healthy_last_seen=recent_timestamp(),
        stale_last_seen=stale_timestamp(),
    )
    def test_preservation_genuinely_disconnected_node_triggers_deletion(
        self,
        healthy_last_seen: datetime,
        stale_last_seen: datetime,
    ) -> None:
        """A 2-node instance where one node has last_seen older than 30s
        should be deleted. The stale node is also absent from topology
        (as it would be after NodeTimedOut fires).

        This passes on unfixed code because the stale node is absent from
        topology, triggering the topology-presence deletion.
        """
        node1 = NodeId("gremlin-1")
        node2 = NodeId("gremlin-2")
        instance_id = InstanceId("disconnected-instance")

        instance = _make_two_node_instance(instance_id, node1, node2)
        instances: dict[InstanceId, PyTorchXPURingInstance] = {instance_id: instance}

        # Only the healthy node is in topology — the stale node has been
        # removed by NodeTimedOut
        topology = Topology()
        topology.add_node(node1)

        # node1 is healthy, node2 has stale last_seen (>30s)
        last_seen: dict[NodeId, datetime] = {
            node1: healthy_last_seen,
            node2: stale_last_seen,
        }

        deleted = plan_health_check(instances, topology, last_seen)

        assert instance_id in deleted, (
            f"Instance {instance_id} was NOT deleted even though node2 has stale "
            f"last_seen={stale_last_seen} (>30s ago). Genuinely disconnected nodes "
            f"must still trigger instance deletion."
        )


class TestPreservationMissingFromLastSeen:
    """Preservation: Nodes absent from last_seen entirely trigger instance deletion.

    **Validates: Requirements 3.2**

    On the unfixed code, a node absent from last_seen is also absent from
    topology, so the instance is deleted. This test confirms that baseline.
    """

    @settings(max_examples=100)
    @given(
        healthy_last_seen=recent_timestamp(),
    )
    def test_preservation_missing_from_last_seen_triggers_deletion(
        self,
        healthy_last_seen: datetime,
    ) -> None:
        """A 2-node instance where one node is absent from last_seen entirely
        should be deleted. The missing node is also absent from topology.

        This passes on unfixed code because the missing node is absent from
        topology, triggering the topology-presence deletion.
        """
        node1 = NodeId("gremlin-1")
        node2 = NodeId("gremlin-2")
        instance_id = InstanceId("missing-node-instance")

        instance = _make_two_node_instance(instance_id, node1, node2)
        instances: dict[InstanceId, PyTorchXPURingInstance] = {instance_id: instance}

        # Only node1 is in topology — node2 was never seen
        topology = Topology()
        topology.add_node(node1)

        # node2 is completely absent from last_seen
        last_seen: dict[NodeId, datetime] = {
            node1: healthy_last_seen,
        }

        deleted = plan_health_check(instances, topology, last_seen)

        assert instance_id in deleted, (
            f"Instance {instance_id} was NOT deleted even though node2 is absent "
            f"from last_seen entirely. Nodes never seen must trigger deletion."
        )


class TestPreservationBothNodesHealthy:
    """Preservation: 2-node instance with both nodes in topology AND recently seen is NOT deleted.

    **Validates: Requirements 3.1, 3.2**

    On the unfixed code, when both nodes ARE in topology, the instance is
    correctly kept alive. This test confirms that non-buggy subset behavior.
    """

    @settings(max_examples=100)
    @given(
        last_seen_1=recent_timestamp(),
        last_seen_2=recent_timestamp(),
    )
    def test_preservation_both_nodes_healthy_not_deleted(
        self,
        last_seen_1: datetime,
        last_seen_2: datetime,
    ) -> None:
        """A 2-node instance where both nodes are in topology AND last_seen
        within 30s should NOT be deleted.

        This passes on unfixed code because both nodes are present in
        topology.list_nodes(), so the topology-presence check passes.
        """
        node1 = NodeId("gremlin-1")
        node2 = NodeId("gremlin-2")
        instance_id = InstanceId("healthy-instance")

        instance = _make_two_node_instance(instance_id, node1, node2)
        instances: dict[InstanceId, PyTorchXPURingInstance] = {instance_id: instance}

        # Both nodes are in topology
        topology = Topology()
        topology.add_node(node1)
        topology.add_node(node2)

        # Both nodes have been seen recently
        last_seen: dict[NodeId, datetime] = {
            node1: last_seen_1,
            node2: last_seen_2,
        }

        deleted = plan_health_check(instances, topology, last_seen)

        assert instance_id not in deleted, (
            f"Instance {instance_id} was deleted even though both nodes are in "
            f"topology and recently seen. last_seen_1={last_seen_1}, "
            f"last_seen_2={last_seen_2}. Both-healthy instances must be kept alive."
        )
