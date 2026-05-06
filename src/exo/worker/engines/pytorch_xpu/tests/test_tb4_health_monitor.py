"""Unit tests for TB4 link health monitoring and graceful degradation.

Tests the TB4LinkHealthMonitor class for periodic link health checking,
link-down/link-up event detection and reporting, and the handle_tp_link_failure()
function for graceful degradation from tensor parallelism to pipeline parallelism.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from exo.worker.engines.pytorch_xpu.tb4_health_monitor import (
    InstanceFailureState,
    LinkEventType,
    TB4LinkEvent,
    TB4LinkHealthMonitor,
    handle_tp_link_failure,
)


@pytest.fixture
def tmp_sys_net(tmp_path: Path) -> Path:
    """Create a temporary /sys/class/net structure with TB4 interfaces."""
    sys_net = tmp_path / "sys" / "class" / "net"
    sys_net.mkdir(parents=True)
    return sys_net


def _create_interface(sys_net: Path, name: str, operstate: str = "up") -> Path:
    """Helper to create a mock network interface directory with operstate."""
    iface_dir = sys_net / name
    iface_dir.mkdir(parents=True, exist_ok=True)
    (iface_dir / "operstate").write_text(operstate + "\n")
    return iface_dir


class TestTB4LinkHealthMonitorInit:
    """Test TB4LinkHealthMonitor initialization and configuration."""

    def test_default_check_interval(self) -> None:
        """Default check interval should be 60 seconds."""
        monitor = TB4LinkHealthMonitor()
        assert monitor.check_interval_seconds == 60.0

    def test_custom_check_interval(self) -> None:
        """Check interval should be configurable."""
        monitor = TB4LinkHealthMonitor(check_interval_seconds=30.0)
        assert monitor.check_interval_seconds == 30.0

    def test_not_running_initially(self) -> None:
        """Monitor should not be running after construction."""
        monitor = TB4LinkHealthMonitor()
        assert not monitor.is_running

    def test_empty_interface_states_initially(self) -> None:
        """Interface states should be empty before first check."""
        monitor = TB4LinkHealthMonitor()
        assert monitor.interface_states == {}


class TestEnumerateTB4Interfaces:
    """Test TB4 interface enumeration from sys_net_path."""

    def test_finds_thunderbolt_interfaces(self, tmp_sys_net: Path) -> None:
        """Should find interfaces starting with 'thunderbolt'."""
        _create_interface(tmp_sys_net, "thunderbolt0")
        _create_interface(tmp_sys_net, "thunderbolt1")
        _create_interface(tmp_sys_net, "eth0")  # Should be ignored

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        interfaces = monitor._enumerate_tb4_interfaces()

        assert "thunderbolt0" in interfaces
        assert "thunderbolt1" in interfaces
        assert "eth0" not in interfaces

    def test_returns_empty_when_no_tb4(self, tmp_sys_net: Path) -> None:
        """Should return empty list when no thunderbolt interfaces exist."""
        _create_interface(tmp_sys_net, "eth0")
        _create_interface(tmp_sys_net, "wlan0")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        interfaces = monitor._enumerate_tb4_interfaces()

        assert interfaces == []

    def test_returns_empty_when_path_missing(self, tmp_path: Path) -> None:
        """Should return empty list when sys_net_path doesn't exist."""
        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_path / "nonexistent")
        interfaces = monitor._enumerate_tb4_interfaces()

        assert interfaces == []

    def test_interfaces_sorted(self, tmp_sys_net: Path) -> None:
        """Interfaces should be returned in sorted order."""
        _create_interface(tmp_sys_net, "thunderbolt2")
        _create_interface(tmp_sys_net, "thunderbolt0")
        _create_interface(tmp_sys_net, "thunderbolt1")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        interfaces = monitor._enumerate_tb4_interfaces()

        assert interfaces == ["thunderbolt0", "thunderbolt1", "thunderbolt2"]


class TestReadInterfaceState:
    """Test reading operstate from interface sysfs."""

    def test_reads_up_state(self, tmp_sys_net: Path) -> None:
        """Should read 'up' from operstate file."""
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        state = monitor._read_interface_state("thunderbolt0")

        assert state == "up"

    def test_reads_down_state(self, tmp_sys_net: Path) -> None:
        """Should read 'down' from operstate file."""
        _create_interface(tmp_sys_net, "thunderbolt0", "down")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        state = monitor._read_interface_state("thunderbolt0")

        assert state == "down"

    def test_strips_whitespace(self, tmp_sys_net: Path) -> None:
        """Should strip trailing newline/whitespace from operstate."""
        iface_dir = tmp_sys_net / "thunderbolt0"
        iface_dir.mkdir(parents=True)
        (iface_dir / "operstate").write_text("up\n\n")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        state = monitor._read_interface_state("thunderbolt0")

        assert state == "up"

    def test_returns_unknown_on_missing_file(self, tmp_sys_net: Path) -> None:
        """Should return 'unknown' when operstate file doesn't exist."""
        (tmp_sys_net / "thunderbolt0").mkdir(parents=True)
        # No operstate file created

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        state = monitor._read_interface_state("thunderbolt0")

        assert state == "unknown"

    def test_returns_unknown_on_missing_interface(self, tmp_sys_net: Path) -> None:
        """Should return 'unknown' when interface directory doesn't exist."""
        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        state = monitor._read_interface_state("thunderbolt99")

        assert state == "unknown"


class TestCheckInterfaces:
    """Test link state change detection.

    Requirements: 11.4, 11.5
    """

    def test_detects_link_down(self, tmp_sys_net: Path) -> None:
        """Should emit LINK_DOWN when interface transitions from up to down.

        Requirements: 11.4
        """
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        # Establish baseline
        monitor._interface_states["thunderbolt0"] = "up"

        # Simulate link going down
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("down\n")

        events = monitor._check_interfaces()

        assert len(events) == 1
        assert events[0].interface_name == "thunderbolt0"
        assert events[0].event_type == LinkEventType.LINK_DOWN
        assert events[0].previous_state == "up"
        assert events[0].current_state == "down"

    def test_detects_link_up(self, tmp_sys_net: Path) -> None:
        """Should emit LINK_UP when interface transitions from down to up.

        Requirements: 11.5
        """
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        # Establish baseline as down
        monitor._interface_states["thunderbolt0"] = "down"

        events = monitor._check_interfaces()

        assert len(events) == 1
        assert events[0].interface_name == "thunderbolt0"
        assert events[0].event_type == LinkEventType.LINK_UP
        assert events[0].previous_state == "down"
        assert events[0].current_state == "up"

    def test_no_event_when_state_unchanged(self, tmp_sys_net: Path) -> None:
        """Should not emit events when state hasn't changed."""
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        monitor._interface_states["thunderbolt0"] = "up"

        events = monitor._check_interfaces()

        assert events == []

    def test_detects_interface_disappearance(self, tmp_sys_net: Path) -> None:
        """Should emit LINK_DOWN when a tracked interface disappears."""
        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        # Interface was tracked as up but no longer exists in filesystem
        monitor._interface_states["thunderbolt0"] = "up"

        events = monitor._check_interfaces()

        assert len(events) == 1
        assert events[0].interface_name == "thunderbolt0"
        assert events[0].event_type == LinkEventType.LINK_DOWN
        assert events[0].current_state == "down"

    def test_multiple_interfaces_independent(self, tmp_sys_net: Path) -> None:
        """Each interface should be tracked independently."""
        _create_interface(tmp_sys_net, "thunderbolt0", "down")
        _create_interface(tmp_sys_net, "thunderbolt1", "up")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        monitor._interface_states["thunderbolt0"] = "up"
        monitor._interface_states["thunderbolt1"] = "up"

        events = monitor._check_interfaces()

        # Only thunderbolt0 should have a link_down event
        assert len(events) == 1
        assert events[0].interface_name == "thunderbolt0"
        assert events[0].event_type == LinkEventType.LINK_DOWN


class TestMonitorLoop:
    """Test the async monitoring loop.

    Requirements: 11.4, 11.5
    """

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_sys_net: Path) -> None:
        """Monitor should start and stop cleanly."""
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
        )

        monitor.start()
        assert monitor.is_running

        # Let it run briefly
        await asyncio.sleep(0.1)

        monitor.stop()
        assert not monitor.is_running

        # Give the task time to finish
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_callback_invoked_on_link_down(self, tmp_sys_net: Path) -> None:
        """Callback should be invoked when link goes down.

        Requirements: 11.4
        """
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        received_events: list[TB4LinkEvent] = []

        def on_event(event: TB4LinkEvent) -> None:
            received_events.append(event)

        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
            callback=on_event,
        )

        monitor.start()

        # Wait for initial scan
        await asyncio.sleep(0.03)

        # Simulate link going down
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("down\n")

        # Wait for next check cycle
        await asyncio.sleep(0.1)

        monitor.stop()
        await asyncio.sleep(0.05)

        # Should have received a link_down event
        link_down_events = [
            e for e in received_events if e.event_type == LinkEventType.LINK_DOWN
        ]
        assert len(link_down_events) >= 1
        assert link_down_events[0].interface_name == "thunderbolt0"

    @pytest.mark.asyncio
    async def test_callback_invoked_on_link_up(self, tmp_sys_net: Path) -> None:
        """Callback should be invoked when link is restored.

        Requirements: 11.5
        """
        # Start with link down
        _create_interface(tmp_sys_net, "thunderbolt0", "down")

        received_events: list[TB4LinkEvent] = []

        def on_event(event: TB4LinkEvent) -> None:
            received_events.append(event)

        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
            callback=on_event,
        )

        monitor.start()

        # Wait for initial scan
        await asyncio.sleep(0.03)

        # Simulate link coming back up
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("up\n")

        # Wait for next check cycle
        await asyncio.sleep(0.1)

        monitor.stop()
        await asyncio.sleep(0.05)

        # Should have received a link_up event
        link_up_events = [
            e for e in received_events if e.event_type == LinkEventType.LINK_UP
        ]
        assert len(link_up_events) >= 1
        assert link_up_events[0].interface_name == "thunderbolt0"

    @pytest.mark.asyncio
    async def test_establishes_baseline_on_start(self, tmp_sys_net: Path) -> None:
        """Monitor should establish baseline state on start without emitting events."""
        _create_interface(tmp_sys_net, "thunderbolt0", "up")
        _create_interface(tmp_sys_net, "thunderbolt1", "down")

        received_events: list[TB4LinkEvent] = []

        def on_event(event: TB4LinkEvent) -> None:
            received_events.append(event)

        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
            callback=on_event,
        )

        monitor.start()

        # Wait for initial scan + one check cycle with no changes
        await asyncio.sleep(0.08)

        monitor.stop()
        await asyncio.sleep(0.05)

        # No events should be emitted during baseline establishment
        # (states haven't changed since initial scan)
        assert received_events == []

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_monitor(
        self, tmp_sys_net: Path
    ) -> None:
        """Monitor should continue running even if callback raises."""
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        def bad_callback(event: TB4LinkEvent) -> None:
            raise ValueError("callback error")

        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
            callback=bad_callback,
        )

        monitor.start()
        await asyncio.sleep(0.03)

        # Trigger an event
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("down\n")
        await asyncio.sleep(0.1)

        # Monitor should still be running despite callback error
        assert monitor.is_running

        monitor.stop()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, tmp_sys_net: Path) -> None:
        """Calling stop() multiple times should not raise."""
        monitor = TB4LinkHealthMonitor(
            check_interval_seconds=0.05,
            sys_net_path=tmp_sys_net,
        )

        monitor.start()
        await asyncio.sleep(0.03)

        monitor.stop()
        monitor.stop()  # Should not raise
        monitor.stop()  # Should not raise


class TestHandleTPLinkFailure:
    """Test graceful degradation on TB4 link failure.

    Requirements: 11.1, 11.2, 11.3
    """

    def test_returns_failed_state(self) -> None:
        """Should return InstanceFailureState with status='failed'.

        Requirements: 11.2
        """
        error = RuntimeError("Gloo transport error: connection reset")
        result = handle_tp_link_failure(
            instance_id="tp-instance-001",
            rank=2,
            error=error,
        )

        assert isinstance(result, InstanceFailureState)
        assert result.status == "failed"

    def test_includes_instance_id(self) -> None:
        """Should include the instance ID in the failure state."""
        error = RuntimeError("all-reduce timeout")
        result = handle_tp_link_failure(
            instance_id="tp-instance-xyz",
            rank=0,
            error=error,
        )

        assert result.instance_id == "tp-instance-xyz"

    def test_includes_failure_reason(self) -> None:
        """Should include a descriptive failure reason.

        Requirements: 11.1
        """
        error = RuntimeError("Gloo all-reduce failed: peer disconnected")
        result = handle_tp_link_failure(
            instance_id="tp-001",
            rank=1,
            error=error,
        )

        assert "rank 1" in result.failure_reason
        assert "RuntimeError" in result.failure_reason
        assert "peer disconnected" in result.failure_reason

    def test_includes_failed_rank(self) -> None:
        """Should record which rank detected the failure."""
        error = RuntimeError("timeout")
        result = handle_tp_link_failure(
            instance_id="tp-001",
            rank=3,
            error=error,
        )

        assert result.failed_rank == 3

    def test_recommends_pipeline_fallback(self) -> None:
        """Should recommend pipeline parallelism as fallback.

        Requirements: 11.3
        """
        error = RuntimeError("TB4 link down")
        result = handle_tp_link_failure(
            instance_id="tp-001",
            rank=0,
            error=error,
        )

        assert result.recommend_pipeline_fallback is True

    def test_handles_different_error_types(self) -> None:
        """Should work with various exception types."""
        error = OSError("Network unreachable")
        result = handle_tp_link_failure(
            instance_id="tp-002",
            rank=1,
            error=error,
        )

        assert result.status == "failed"
        assert "OSError" in result.failure_reason
        assert "Network unreachable" in result.failure_reason


class TestGracefulDegradationFlow:
    """Test the full graceful degradation flow: TP failure → PP fallback.

    This tests the integration between link failure detection and the
    state transition that enables the master to re-place using pipeline
    parallelism.

    Requirements: 11.1, 11.2, 11.3
    """

    def test_allreduce_failure_produces_failed_state(self) -> None:
        """Simulates: all-reduce fails → handle_tp_link_failure → failed state.

        This is the core degradation path:
        1. TensorParallelShard._all_reduce() raises RuntimeError
        2. Generation pipeline catches it
        3. handle_tp_link_failure() produces InstanceFailureState
        4. Master can use this to trigger pipeline parallelism re-placement
        """
        # Simulate the error that would come from a failed all-reduce
        allreduce_error = RuntimeError(
            "Tensor-parallel all-reduce failed: layer_index=5, "
            "tensor_shape=(1, 1, 2560), timeout=30s, rank=2"
        )

        # This is what the generation pipeline would call
        failure_state = handle_tp_link_failure(
            instance_id="tp-qwen35-4b",
            rank=2,
            error=allreduce_error,
        )

        # Verify the state enables PP fallback
        assert failure_state.status == "failed"
        assert failure_state.recommend_pipeline_fallback is True
        assert failure_state.instance_id == "tp-qwen35-4b"
        assert failure_state.failed_rank == 2
        assert "all-reduce failed" in failure_state.failure_reason

    def test_link_down_event_followed_by_failure_state(
        self, tmp_sys_net: Path
    ) -> None:
        """Simulates: link goes down → monitor detects → generation fails → PP fallback.

        Full flow:
        1. TB4LinkHealthMonitor detects link_down
        2. Next all-reduce in generation fails
        3. handle_tp_link_failure() produces failed state
        4. Master re-places with pipeline parallelism
        """
        _create_interface(tmp_sys_net, "thunderbolt0", "up")

        # Step 1: Monitor detects link down
        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        monitor._interface_states["thunderbolt0"] = "up"

        # Simulate link going down
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("down\n")
        events = monitor._check_interfaces()

        assert len(events) == 1
        assert events[0].event_type == LinkEventType.LINK_DOWN

        # Step 2: Generation fails due to communication error
        comm_error = RuntimeError(
            "Gloo transport error: connection reset by peer"
        )

        # Step 3: Produce failure state
        failure_state = handle_tp_link_failure(
            instance_id="tp-instance-001",
            rank=0,
            error=comm_error,
        )

        # Step 4: Verify master can use this for PP fallback
        assert failure_state.status == "failed"
        assert failure_state.recommend_pipeline_fallback is True

    def test_link_restored_after_failure(self, tmp_sys_net: Path) -> None:
        """After failure and PP fallback, link restoration enables future TP.

        Requirements: 11.5
        """
        _create_interface(tmp_sys_net, "thunderbolt0", "down")

        monitor = TB4LinkHealthMonitor(sys_net_path=tmp_sys_net)
        monitor._interface_states["thunderbolt0"] = "down"

        # Simulate link coming back up
        (tmp_sys_net / "thunderbolt0" / "operstate").write_text("up\n")
        events = monitor._check_interfaces()

        assert len(events) == 1
        assert events[0].event_type == LinkEventType.LINK_UP
        assert events[0].interface_name == "thunderbolt0"

        # After link restoration, the master could consider TP for future placements


class TestTB4LinkEvent:
    """Test TB4LinkEvent dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TB4LinkEvent should be immutable (frozen)."""
        event = TB4LinkEvent(
            interface_name="thunderbolt0",
            event_type=LinkEventType.LINK_DOWN,
            previous_state="up",
            current_state="down",
        )

        with pytest.raises(AttributeError):
            event.interface_name = "thunderbolt1"  # type: ignore[misc]

    def test_link_down_event_fields(self) -> None:
        """Link down event should have correct fields."""
        event = TB4LinkEvent(
            interface_name="thunderbolt1",
            event_type=LinkEventType.LINK_DOWN,
            previous_state="up",
            current_state="down",
        )

        assert event.interface_name == "thunderbolt1"
        assert event.event_type == LinkEventType.LINK_DOWN
        assert event.previous_state == "up"
        assert event.current_state == "down"

    def test_link_up_event_fields(self) -> None:
        """Link up event should have correct fields."""
        event = TB4LinkEvent(
            interface_name="thunderbolt0",
            event_type=LinkEventType.LINK_UP,
            previous_state="down",
            current_state="up",
        )

        assert event.interface_name == "thunderbolt0"
        assert event.event_type == LinkEventType.LINK_UP
        assert event.previous_state == "down"
        assert event.current_state == "up"


class TestInstanceFailureState:
    """Test InstanceFailureState dataclass."""

    def test_frozen_dataclass(self) -> None:
        """InstanceFailureState should be immutable (frozen)."""
        state = InstanceFailureState(
            instance_id="tp-001",
            failure_reason="test",
            failed_rank=0,
        )

        with pytest.raises(AttributeError):
            state.instance_id = "tp-002"  # type: ignore[misc]

    def test_default_status_is_failed(self) -> None:
        """Default status should be 'failed'."""
        state = InstanceFailureState(
            instance_id="tp-001",
            failure_reason="test",
            failed_rank=0,
        )

        assert state.status == "failed"

    def test_default_recommends_pipeline_fallback(self) -> None:
        """Default should recommend pipeline fallback."""
        state = InstanceFailureState(
            instance_id="tp-001",
            failure_reason="test",
            failed_rank=0,
        )

        assert state.recommend_pipeline_fallback is True
