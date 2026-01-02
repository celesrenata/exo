"""
Shutdown coordination infrastructure for preventing race conditions in multiprocessing environments.

This module implements a three-phase shutdown protocol to ensure graceful termination of runner processes
without encountering "Queue is closed" or "ClosedResourceError" exceptions.
"""

import asyncio
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from uuid import uuid4

from loguru import logger


class ShutdownPhase(Enum):
    """Phases of the three-phase shutdown protocol."""

    SIGNALING = "signaling"  # Notify all processes of shutdown intent
    DRAINING = "draining"  # Drain all queues and channels
    CLOSING = "closing"  # Close resources in dependency order
    COMPLETE = "complete"  # All resources cleaned up
    FAILED = "failed"  # Shutdown failed, force cleanup


@dataclass
class ShutdownState:
    """State tracking for shutdown coordination."""

    phase: ShutdownPhase
    started_at: datetime
    timeout_at: datetime
    runner_id: str
    resources_remaining: Set[str] = field(default_factory=set)
    error_count: int = 0
    last_error: Optional[Exception] = None


class ShutdownCoordinator:
    """
    Coordinates graceful shutdown across runner processes using a three-phase protocol.

    The three phases are:
    1. SIGNALING: Notify all processes of shutdown intent
    2. DRAINING: Drain all queues and channels
    3. CLOSING: Close resources in dependency order
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the shutdown coordinator.

        Args:
            temp_dir: Directory for coordination files. If None, uses system temp directory.
        """
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._coordination_dir = self._temp_dir / "exo_shutdown_coordination"
        self._coordination_dir.mkdir(exist_ok=True)

        # In-process state tracking
        self._shutdown_states: Dict[str, ShutdownState] = {}
        self._shutdown_handlers: List[Callable[[str], None]] = []
        self._lock = asyncio.Lock()

        # Cross-process coordination using file-based signaling
        self._signal_files: Dict[str, Path] = {}

    async def initiate_shutdown(self, runner_id: str, timeout: float = 30.0) -> bool:
        """
        Initiate graceful shutdown for a runner process.

        Args:
            runner_id: Unique identifier for the runner
            timeout: Maximum time to wait for shutdown completion

        Returns:
            True if shutdown completed successfully, False if timeout or error
        """
        async with self._lock:
            if runner_id in self._shutdown_states:
                logger.warning(f"Shutdown already in progress for runner {runner_id}")
                return await self._wait_for_existing_shutdown(runner_id)

            # Create shutdown state
            now = datetime.now()
            timeout_at = now + timedelta(seconds=timeout)

            shutdown_state = ShutdownState(
                phase=ShutdownPhase.SIGNALING,
                started_at=now,
                timeout_at=timeout_at,
                runner_id=runner_id,
            )

            self._shutdown_states[runner_id] = shutdown_state

            # Create coordination file for cross-process signaling
            signal_file = (
                self._coordination_dir
                / f"shutdown_{runner_id}_{uuid4().hex[:8]}.signal"
            )
            self._signal_files[runner_id] = signal_file

            logger.info(
                f"Initiating shutdown for runner {runner_id} with timeout {timeout}s"
            )

        try:
            # Execute three-phase shutdown
            success = await self._execute_three_phase_shutdown(runner_id, timeout)
            return success

        except Exception as e:
            logger.error(f"Shutdown failed for runner {runner_id}: {e}")
            async with self._lock:
                if runner_id in self._shutdown_states:
                    self._shutdown_states[runner_id].phase = ShutdownPhase.FAILED
                    self._shutdown_states[runner_id].last_error = e
                    self._shutdown_states[runner_id].error_count += 1
            return False

        finally:
            # Cleanup coordination files
            await self._cleanup_coordination_files(runner_id)

    async def _execute_three_phase_shutdown(
        self, runner_id: str, timeout: float
    ) -> bool:
        """Execute the three-phase shutdown protocol."""

        # Phase 1: SIGNALING
        logger.debug(f"Phase 1 (SIGNALING) for runner {runner_id}")
        if not await self._phase_signaling(runner_id):
            return False

        # Phase 2: DRAINING
        logger.debug(f"Phase 2 (DRAINING) for runner {runner_id}")
        if not await self._phase_draining(runner_id):
            return False

        # Phase 3: CLOSING
        logger.debug(f"Phase 3 (CLOSING) for runner {runner_id}")
        if not await self._phase_closing(runner_id):
            return False

        # Mark as complete
        async with self._lock:
            if runner_id in self._shutdown_states:
                self._shutdown_states[runner_id].phase = ShutdownPhase.COMPLETE

        logger.info(f"Shutdown completed successfully for runner {runner_id}")
        return True

    async def _phase_signaling(self, runner_id: str) -> bool:
        """
        Phase 1: Signal shutdown intent to all processes.

        This phase creates coordination files and notifies all registered handlers
        that shutdown is beginning.
        """
        async with self._lock:
            if runner_id not in self._shutdown_states:
                return False

            state = self._shutdown_states[runner_id]
            state.phase = ShutdownPhase.SIGNALING

            # Create signal file for cross-process coordination
            signal_file = self._signal_files.get(runner_id)
            if signal_file:
                try:
                    signal_file.write_text(f"SHUTDOWN_SIGNALING:{time.time()}")
                    logger.debug(f"Created shutdown signal file: {signal_file}")
                except Exception as e:
                    logger.error(f"Failed to create signal file: {e}")
                    state.error_count += 1
                    state.last_error = e
                    return False

            # Notify all registered handlers
            for handler in self._shutdown_handlers:
                try:
                    handler(runner_id)
                except Exception as e:
                    logger.error(f"Shutdown handler failed: {e}")
                    state.error_count += 1
                    state.last_error = e

        # Brief pause to allow signal propagation
        await asyncio.sleep(0.1)
        return True

    async def _phase_draining(self, runner_id: str) -> bool:
        """
        Phase 2: Drain all queues and channels.

        This phase ensures all pending messages are processed before
        closing resources.
        """
        async with self._lock:
            if runner_id not in self._shutdown_states:
                return False

            state = self._shutdown_states[runner_id]
            state.phase = ShutdownPhase.DRAINING

            # Update signal file
            signal_file = self._signal_files.get(runner_id)
            if signal_file:
                try:
                    signal_file.write_text(f"SHUTDOWN_DRAINING:{time.time()}")
                except Exception as e:
                    logger.error(f"Failed to update signal file: {e}")
                    state.error_count += 1
                    state.last_error = e

        # Allow time for draining - this will be enhanced when ResourceManager is implemented
        drain_timeout = min(
            5.0, (state.timeout_at - datetime.now()).total_seconds() / 2
        )
        if drain_timeout > 0:
            await asyncio.sleep(drain_timeout)

        return True

    async def _phase_closing(self, runner_id: str) -> bool:
        """
        Phase 3: Close resources in dependency order.

        This phase performs the actual resource cleanup in the correct order
        to prevent race conditions.
        """
        async with self._lock:
            if runner_id not in self._shutdown_states:
                return False

            state = self._shutdown_states[runner_id]
            state.phase = ShutdownPhase.CLOSING

            # Update signal file
            signal_file = self._signal_files.get(runner_id)
            if signal_file:
                try:
                    signal_file.write_text(f"SHUTDOWN_CLOSING:{time.time()}")
                except Exception as e:
                    logger.error(f"Failed to update signal file: {e}")
                    state.error_count += 1
                    state.last_error = e

        # Resource cleanup will be handled by ResourceManager when implemented
        # For now, just ensure we don't exceed timeout
        remaining_time = (state.timeout_at - datetime.now()).total_seconds()
        if remaining_time <= 0:
            logger.warning(f"Shutdown timeout exceeded for runner {runner_id}")
            return False

        return True

    async def wait_for_shutdown_complete(self, runner_id: str) -> bool:
        """
        Wait for shutdown to complete for the specified runner.

        Args:
            runner_id: Runner to wait for

        Returns:
            True if shutdown completed successfully, False if failed or timeout
        """
        while True:
            async with self._lock:
                if runner_id not in self._shutdown_states:
                    return False

                state = self._shutdown_states[runner_id]

                if state.phase == ShutdownPhase.COMPLETE:
                    return True
                elif state.phase == ShutdownPhase.FAILED:
                    return False
                elif datetime.now() > state.timeout_at:
                    logger.warning(f"Shutdown timeout for runner {runner_id}")
                    state.phase = ShutdownPhase.FAILED
                    return False

            # Brief pause before checking again
            await asyncio.sleep(0.1)

    def register_shutdown_handler(self, handler: Callable[[str], None]) -> None:
        """
        Register a handler to be called during shutdown signaling phase.

        Args:
            handler: Function to call with runner_id during shutdown
        """
        self._shutdown_handlers.append(handler)
        logger.debug(f"Registered shutdown handler: {handler}")

    def get_shutdown_status(self, runner_id: str) -> Optional[ShutdownState]:
        """
        Get the current shutdown status for a runner.

        Args:
            runner_id: Runner to check status for

        Returns:
            ShutdownState if runner is in shutdown process, None otherwise
        """
        return self._shutdown_states.get(runner_id)

    def is_shutdown_in_progress(self, runner_id: str) -> bool:
        """
        Check if shutdown is currently in progress for a runner.

        Args:
            runner_id: Runner to check

        Returns:
            True if shutdown is in progress, False otherwise
        """
        state = self._shutdown_states.get(runner_id)
        return state is not None and state.phase not in (
            ShutdownPhase.COMPLETE,
            ShutdownPhase.FAILED,
        )

    def check_shutdown_signal(self, runner_id: str) -> Optional[ShutdownPhase]:
        """
        Check for shutdown signals from other processes (cross-process coordination).

        Args:
            runner_id: Runner to check signals for

        Returns:
            Current shutdown phase if signal exists, None otherwise
        """
        # Look for any signal files that might apply to this runner
        pattern = f"shutdown_{runner_id}_*.signal"
        for signal_file in self._coordination_dir.glob(pattern):
            try:
                content = signal_file.read_text().strip()
                if content.startswith("SHUTDOWN_"):
                    phase_name = content.split(":")[0].replace("SHUTDOWN_", "")
                    return ShutdownPhase(phase_name.lower())
            except Exception as e:
                logger.debug(f"Error reading signal file {signal_file}: {e}")
                continue

        return None

    async def _wait_for_existing_shutdown(self, runner_id: str) -> bool:
        """Wait for an existing shutdown process to complete."""
        logger.debug(f"Waiting for existing shutdown of runner {runner_id}")
        return await self.wait_for_shutdown_complete(runner_id)

    async def _cleanup_coordination_files(self, runner_id: str) -> None:
        """Clean up coordination files for a runner."""
        try:
            signal_file = self._signal_files.pop(runner_id, None)
            if signal_file and signal_file.exists():
                signal_file.unlink()
                logger.debug(f"Cleaned up signal file: {signal_file}")
        except Exception as e:
            logger.debug(f"Error cleaning up coordination files: {e}")

        # Remove from in-memory state
        async with self._lock:
            self._shutdown_states.pop(runner_id, None)

    async def cleanup_all(self) -> None:
        """Clean up all coordination state and files."""
        async with self._lock:
            # Clean up all signal files
            for runner_id in list(self._signal_files.keys()):
                await self._cleanup_coordination_files(runner_id)

            # Clear handlers
            self._shutdown_handlers.clear()

            # Remove coordination directory if empty
            try:
                if self._coordination_dir.exists() and not any(
                    self._coordination_dir.iterdir()
                ):
                    self._coordination_dir.rmdir()
            except Exception as e:
                logger.debug(f"Error removing coordination directory: {e}")


# Global instance for cross-module usage
_global_coordinator: Optional[ShutdownCoordinator] = None


def get_shutdown_coordinator() -> ShutdownCoordinator:
    """Get the global shutdown coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = ShutdownCoordinator()
    return _global_coordinator


def reset_shutdown_coordinator() -> None:
    """Reset the global shutdown coordinator (mainly for testing)."""
    global _global_coordinator
    _global_coordinator = None
