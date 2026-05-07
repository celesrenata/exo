"""
Thunderbolt 4 Link Health Monitor

Provides periodic health checking of TB4 network interfaces and graceful
degradation handling when TB4 links fail during tensor-parallel generation.

The TB4LinkHealthMonitor runs as an async task, periodically reading
/sys/class/net/thunderboltN/operstate to detect link-down and link-up events.
Events are reported via a callback so the cluster state can be updated.

The handle_tp_link_failure() function provides a simple state transition for
when a tensor-parallel instance fails due to TB4 communication errors, enabling
the master to re-place the model using pipeline parallelism over ethernet.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)


class LinkEventType(Enum):
    """Type of TB4 link event."""

    LINK_DOWN = "link_down"
    LINK_UP = "link_up"


@dataclass(frozen=True)
class TB4LinkEvent:
    """A TB4 link state change event.

    Emitted when a TB4 interface transitions between up and down states.

    Requirements: 11.4, 11.5
    """

    interface_name: str
    """The TB4 network interface that changed state (e.g., 'thunderbolt0')."""

    event_type: LinkEventType
    """Whether the link went down or came back up."""

    previous_state: str
    """The previous operstate value (e.g., 'up', 'down', 'unknown')."""

    current_state: str
    """The current operstate value."""


@dataclass(frozen=True)
class InstanceFailureState:
    """State representing a failed tensor-parallel instance.

    Returned by handle_tp_link_failure() to indicate that the instance
    has failed and the master should consider re-placement using pipeline
    parallelism over ethernet.

    Requirements: 11.2, 11.3
    """

    instance_id: str
    """The ID of the failed instance."""

    failure_reason: str
    """Human-readable description of why the instance failed."""

    failed_rank: int
    """The rank that detected the failure."""

    status: Literal["failed"] = "failed"
    """Always 'failed' for this state."""

    recommend_pipeline_fallback: bool = True
    """Whether the master should attempt pipeline parallelism as fallback."""


# Type alias for the event callback
LinkEventCallback = Callable[[TB4LinkEvent], None]


@dataclass
class TB4LinkHealthMonitor:
    """Periodic health checker for Thunderbolt 4 network interfaces.

    Reads /sys/class/net/thunderboltN/operstate at a configurable interval
    to detect link-down and link-up events. Events are reported via a
    callback function.

    The monitor tracks the last known state of each interface and only
    emits events on state transitions (not on every check).

    Requirements: 11.4, 11.5
    """

    check_interval_seconds: float = 60.0
    """How often to check TB4 interface status (seconds)."""

    sys_net_path: Path = field(default_factory=lambda: Path("/sys/class/net"))
    """Path to /sys/class/net (overridable for testing)."""

    callback: LinkEventCallback | None = None
    """Callback invoked on link state changes."""

    _interface_states: dict[str, str] = field(default_factory=dict, init=False)
    """Last known operstate for each TB4 interface."""

    _running: bool = field(default=False, init=False)
    """Whether the monitor loop is currently running."""

    _task: asyncio.Task[None] | None = field(default=None, init=False)
    """The asyncio task running the monitor loop."""

    def _read_interface_state(self, interface_name: str) -> str:
        """Read the operstate of a TB4 interface.

        Args:
            interface_name: Name of the interface (e.g., 'thunderbolt0').

        Returns:
            The operstate string ('up', 'down', 'unknown', etc.),
            or 'unknown' if the file cannot be read.
        """
        operstate_path = self.sys_net_path / interface_name / "operstate"
        try:
            return operstate_path.read_text().strip()
        except (OSError, IOError):
            return "unknown"

    def _enumerate_tb4_interfaces(self) -> list[str]:
        """Find all thunderbolt* interfaces in sys_net_path.

        Returns:
            List of interface names starting with 'thunderbolt'.
        """
        if not self.sys_net_path.exists():
            return []

        interfaces: list[str] = []
        try:
            for entry in sorted(self.sys_net_path.iterdir()):
                if entry.name.startswith("thunderbolt"):
                    interfaces.append(entry.name)
        except (OSError, IOError):
            pass

        return interfaces

    def _check_interfaces(self) -> list[TB4LinkEvent]:
        """Check all TB4 interfaces and return any state change events.

        Compares current operstate against last known state for each
        interface. Only emits events on transitions.

        Returns:
            List of TB4LinkEvent objects for any state changes detected.
        """
        events: list[TB4LinkEvent] = []
        interfaces = self._enumerate_tb4_interfaces()

        for iface in interfaces:
            current_state = self._read_interface_state(iface)
            previous_state = self._interface_states.get(iface, "unknown")

            if current_state != previous_state:
                # State changed — determine event type
                if current_state in ("up", "unknown"):
                    # Link came up (or became detectable)
                    if previous_state == "down":
                        event = TB4LinkEvent(
                            interface_name=iface,
                            event_type=LinkEventType.LINK_UP,
                            previous_state=previous_state,
                            current_state=current_state,
                        )
                        events.append(event)
                        logger.info(
                            f"TB4 link restored: {iface} "
                            f"({previous_state} → {current_state})"
                        )
                elif current_state == "down":
                    # Link went down
                    event = TB4LinkEvent(
                        interface_name=iface,
                        event_type=LinkEventType.LINK_DOWN,
                        previous_state=previous_state,
                        current_state=current_state,
                    )
                    events.append(event)
                    logger.warning(
                        f"TB4 link down: {iface} "
                        f"({previous_state} → {current_state})"
                    )

                # Update tracked state
                self._interface_states[iface] = current_state

        # Also detect interfaces that disappeared (were tracked but no longer exist)
        tracked_interfaces = set(self._interface_states.keys())
        current_interfaces = set(interfaces)
        disappeared = tracked_interfaces - current_interfaces

        for iface in disappeared:
            prev = self._interface_states.pop(iface)
            if prev != "down":
                event = TB4LinkEvent(
                    interface_name=iface,
                    event_type=LinkEventType.LINK_DOWN,
                    previous_state=prev,
                    current_state="down",
                )
                events.append(event)
                logger.warning(f"TB4 interface disappeared: {iface}")

        return events

    async def _monitor_loop(self) -> None:
        """Main monitoring loop. Runs until stopped."""
        logger.info(
            f"TB4 link health monitor started "
            f"(interval={self.check_interval_seconds}s)"
        )

        # Initial scan to establish baseline state
        interfaces = self._enumerate_tb4_interfaces()
        for iface in interfaces:
            self._interface_states[iface] = self._read_interface_state(iface)

        logger.debug(
            f"Initial TB4 interface states: {self._interface_states}"
        )

        while self._running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            events = self._check_interfaces()

            # Invoke callback for each event
            if self.callback is not None:
                for event in events:
                    try:
                        self.callback(event)
                    except Exception as exc:
                        logger.error(
                            f"Error in link event callback: {exc}",
                            exc_info=True,
                        )

        logger.info("TB4 link health monitor stopped")

    def start(self) -> None:
        """Start the health monitor as an async task.

        Must be called from within a running event loop.
        """
        if self._running:
            logger.warning("TB4 link health monitor already running")
            return

        self._running = True
        self._task = asyncio.ensure_future(self._monitor_loop())

    def stop(self) -> None:
        """Stop the health monitor.

        Cancels the monitoring task. Safe to call multiple times.
        """
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None

    @property
    def is_running(self) -> bool:
        """Whether the monitor is currently running."""
        return self._running

    @property
    def interface_states(self) -> dict[str, str]:
        """Current known states of all tracked TB4 interfaces.

        Returns a copy to prevent external mutation.
        """
        return dict(self._interface_states)


def handle_tp_link_failure(
    instance_id: str,
    rank: int,
    error: Exception,
) -> InstanceFailureState:
    """Handle a tensor-parallel link failure by transitioning to failed state.

    Called when a TB4 all-reduce operation fails due to a communication error
    during generation. Returns an InstanceFailureState that the master can use
    to trigger re-placement using pipeline parallelism over ethernet.

    This is a pure state transition function — it does not perform any I/O
    or modify global state. The caller is responsible for:
    1. Terminating the current generation (yielding error response)
    2. Reporting the failure state to the master
    3. Cleaning up process groups

    Args:
        instance_id: The ID of the tensor-parallel instance that failed.
        rank: The rank that detected the failure.
        error: The exception that caused the failure (e.g., RuntimeError
            from a failed all-reduce).

    Returns:
        InstanceFailureState indicating the instance has failed and
        recommending pipeline parallelism fallback.

    Requirements: 11.1, 11.2, 11.3
    """
    failure_reason = (
        f"TB4 tensor-parallel communication failure on rank {rank}: "
        f"{type(error).__name__}: {error}"
    )

    logger.error(failure_reason)

    return InstanceFailureState(
        instance_id=instance_id,
        failure_reason=failure_reason,
        failed_rank=rank,
        status="failed",
        recommend_pipeline_fallback=True,
    )
