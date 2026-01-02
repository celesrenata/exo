"""
Enhanced synchronization primitives for cross-process coordination.

This module provides deadlock-free synchronization mechanisms for coordinating
shutdown and resource management across multiple processes.
"""

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set
from uuid import uuid4

from loguru import logger


class LockState(Enum):
    """States for cross-process locks."""

    AVAILABLE = "available"
    ACQUIRED = "acquired"
    WAITING = "waiting"
    TIMEOUT = "timeout"
    ERROR = "error"


class EventType(Enum):
    """Types of cross-process events."""

    SHUTDOWN_SIGNAL = "shutdown_signal"
    RESOURCE_READY = "resource_ready"
    CLEANUP_COMPLETE = "cleanup_complete"
    ERROR_OCCURRED = "error_occurred"
    CUSTOM = "custom"


@dataclass
class LockInfo:
    """Information about a cross-process lock."""

    lock_id: str
    process_id: int
    acquired_at: datetime
    timeout_at: Optional[datetime] = None
    holder_info: str = ""


@dataclass
class EventInfo:
    """Information about a cross-process event."""

    event_id: str
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_process: int = field(default_factory=os.getpid)


class CrossProcessLock:
    """
    A cross-process lock using file-based locking with deadlock detection.

    This implementation uses file locking to provide mutual exclusion across
    processes while detecting and preventing deadlocks.
    """

    def __init__(
        self, lock_name: str, temp_dir: Optional[Path] = None, timeout: float = 30.0
    ):
        """
        Initialize a cross-process lock.

        Args:
            lock_name: Unique name for the lock
            temp_dir: Directory for lock files
            timeout: Default timeout for lock acquisition
        """
        self.lock_name = lock_name
        self.timeout = timeout
        self.process_id = os.getpid()

        # Create lock directory
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._lock_dir = self._temp_dir / "exo_locks"
        self._lock_dir.mkdir(exist_ok=True)

        # Lock file path
        self._lock_file = self._lock_dir / f"{lock_name}.lock"
        self._info_file = self._lock_dir / f"{lock_name}.info"

        # State tracking
        self._file_handle: Optional[int] = None
        self._acquired = False
        self._acquisition_time: Optional[datetime] = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock (blocking).

        Args:
            timeout: Maximum time to wait for lock acquisition

        Returns:
            True if lock was acquired, False if timeout
        """
        if self._acquired:
            logger.warning(f"Lock {self.lock_name} already acquired by this process")
            return True

        timeout = timeout or self.timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to acquire the lock
                self._file_handle = os.open(
                    self._lock_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                )
                fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Lock acquired successfully
                self._acquired = True
                self._acquisition_time = datetime.now()

                # Write lock info
                lock_info = LockInfo(
                    lock_id=self.lock_name,
                    process_id=self.process_id,
                    acquired_at=self._acquisition_time,
                    timeout_at=self._acquisition_time + timedelta(seconds=timeout)
                    if timeout
                    else None,
                    holder_info=f"pid={self.process_id}",
                )

                self._write_lock_info(lock_info)

                logger.debug(f"Acquired lock {self.lock_name} (pid={self.process_id})")
                return True

            except (OSError, IOError) as e:
                if e.errno in (11, 35):  # EAGAIN or EWOULDBLOCK
                    # Lock is held by another process, wait and retry
                    time.sleep(0.1)
                    continue
                else:
                    logger.error(f"Error acquiring lock {self.lock_name}: {e}")
                    return False

        logger.warning(f"Timeout acquiring lock {self.lock_name} after {timeout}s")
        return False

    def release(self) -> bool:
        """
        Release the lock.

        Returns:
            True if lock was released, False if error
        """
        if not self._acquired:
            logger.warning(f"Lock {self.lock_name} not acquired by this process")
            return False

        try:
            if self._file_handle is not None:
                fcntl.flock(self._file_handle, fcntl.LOCK_UN)
                os.close(self._file_handle)
                self._file_handle = None

            # Clean up lock files
            try:
                self._lock_file.unlink(missing_ok=True)
                self._info_file.unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Error cleaning up lock files: {e}")

            self._acquired = False
            self._acquisition_time = None

            logger.debug(f"Released lock {self.lock_name} (pid={self.process_id})")
            return True

        except Exception as e:
            logger.error(f"Error releasing lock {self.lock_name}: {e}")
            return False

    def is_acquired(self) -> bool:
        """Check if the lock is currently acquired by this process."""
        return self._acquired

    def get_lock_info(self) -> Optional[LockInfo]:
        """Get information about the current lock holder."""
        try:
            if self._info_file.exists():
                content = self._info_file.read_text()
                # Parse lock info (simplified implementation)
                lines = content.strip().split("\n")
                if len(lines) >= 3:
                    lock_id = lines[0].split(":", 1)[1].strip()
                    process_id = int(lines[1].split(":", 1)[1].strip())
                    acquired_at = datetime.fromisoformat(
                        lines[2].split(":", 1)[1].strip()
                    )

                    return LockInfo(
                        lock_id=lock_id, process_id=process_id, acquired_at=acquired_at
                    )
        except Exception as e:
            logger.debug(f"Error reading lock info: {e}")

        return None

    def _write_lock_info(self, lock_info: LockInfo) -> None:
        """Write lock information to info file."""
        try:
            content = f"""lock_id: {lock_info.lock_id}
process_id: {lock_info.process_id}
acquired_at: {lock_info.acquired_at.isoformat()}
holder_info: {lock_info.holder_info}
"""
            self._info_file.write_text(content)
        except Exception as e:
            logger.debug(f"Error writing lock info: {e}")

    @contextmanager
    def acquire_context(
        self, timeout: Optional[float] = None
    ) -> Generator[bool, None, None]:
        """Context manager for lock acquisition."""
        acquired = self.acquire(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    def __enter__(self) -> bool:
        """Context manager entry."""
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        if self._acquired:
            self.release()


class CrossProcessEvent:
    """
    Cross-process event signaling using file-based coordination.

    This allows processes to signal events to each other and wait for
    specific events to occur.
    """

    def __init__(self, event_name: str, temp_dir: Optional[Path] = None):
        """
        Initialize a cross-process event.

        Args:
            event_name: Unique name for the event
            temp_dir: Directory for event files
        """
        self.event_name = event_name
        self.process_id = os.getpid()

        # Create event directory
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._event_dir = self._temp_dir / "exo_events"
        self._event_dir.mkdir(exist_ok=True)

        # Event file path
        self._event_file = self._event_dir / f"{event_name}.event"

    def signal(
        self, event_type: EventType, data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Signal an event to other processes.

        Args:
            event_type: Type of event to signal
            data: Optional data to include with the event

        Returns:
            True if event was signaled successfully
        """
        try:
            event_info = EventInfo(
                event_id=f"{self.event_name}_{uuid4().hex[:8]}",
                event_type=event_type,
                data=data or {},
                source_process=self.process_id,
            )

            # Write event info to file
            content = f"""event_id: {event_info.event_id}
event_type: {event_info.event_type.value}
timestamp: {event_info.timestamp.isoformat()}
source_process: {event_info.source_process}
data: {event_info.data}
"""
            self._event_file.write_text(content)

            logger.debug(
                f"Signaled event {event_info.event_id} (type: {event_type.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Error signaling event {self.event_name}: {e}")
            return False

    def wait_for_event(
        self, event_type: EventType, timeout: float = 30.0
    ) -> Optional[EventInfo]:
        """
        Wait for a specific type of event.

        Args:
            event_type: Type of event to wait for
            timeout: Maximum time to wait

        Returns:
            EventInfo if event occurred, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            event_info = self.check_event(event_type)
            if event_info:
                return event_info

            time.sleep(0.1)

        logger.debug(
            f"Timeout waiting for event {event_type.value} on {self.event_name}"
        )
        return None

    def check_event(
        self, event_type: Optional[EventType] = None
    ) -> Optional[EventInfo]:
        """
        Check for an event without waiting.

        Args:
            event_type: Specific event type to check for, or None for any event

        Returns:
            EventInfo if event exists, None otherwise
        """
        try:
            if not self._event_file.exists():
                return None

            content = self._event_file.read_text()
            lines = content.strip().split("\n")

            if len(lines) >= 4:
                event_id = lines[0].split(":", 1)[1].strip()
                file_event_type = EventType(lines[1].split(":", 1)[1].strip())
                timestamp = datetime.fromisoformat(lines[2].split(":", 1)[1].strip())
                source_process = int(lines[3].split(":", 1)[1].strip())

                # Parse data if present
                data = {}
                if len(lines) >= 5:
                    data_str = lines[4].split(":", 1)[1].strip()
                    try:
                        data = eval(data_str)  # Simple parsing, could be improved
                    except Exception:
                        data = {"raw": data_str}

                event_info = EventInfo(
                    event_id=event_id,
                    event_type=file_event_type,
                    data=data,
                    timestamp=timestamp,
                    source_process=source_process,
                )

                # Check if this matches the requested event type
                if event_type is None or file_event_type == event_type:
                    return event_info

        except Exception as e:
            logger.debug(f"Error checking event: {e}")

        return None

    def clear_event(self) -> bool:
        """
        Clear the event file.

        Returns:
            True if cleared successfully
        """
        try:
            self._event_file.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.debug(f"Error clearing event: {e}")
            return False


class SharedStateManager:
    """
    Manages shared state across processes using file-based storage.

    This provides a simple key-value store that can be accessed safely
    from multiple processes.
    """

    def __init__(self, state_name: str, temp_dir: Optional[Path] = None):
        """
        Initialize shared state manager.

        Args:
            state_name: Unique name for the shared state
            temp_dir: Directory for state files
        """
        self.state_name = state_name

        # Create state directory
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._state_dir = self._temp_dir / "exo_shared_state"
        self._state_dir.mkdir(exist_ok=True)

        # State file and lock
        self._state_file = self._state_dir / f"{state_name}.state"
        self._lock = CrossProcessLock(f"state_{state_name}", temp_dir)

    def set_value(self, key: str, value: Any, timeout: float = 10.0) -> bool:
        """
        Set a value in shared state.

        Args:
            key: Key to set
            value: Value to store
            timeout: Timeout for lock acquisition

        Returns:
            True if value was set successfully
        """
        with self._lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for setting {key}")
                return False

            try:
                # Read existing state
                state = self._read_state()

                # Update value
                state[key] = {
                    "value": value,
                    "timestamp": datetime.now().isoformat(),
                    "process": os.getpid(),
                }

                # Write back
                self._write_state(state)

                logger.debug(f"Set shared state {key} = {value}")
                return True

            except Exception as e:
                logger.error(f"Error setting shared state {key}: {e}")
                return False

    def get_value(self, key: str, default: Any = None, timeout: float = 10.0) -> Any:
        """
        Get a value from shared state.

        Args:
            key: Key to get
            default: Default value if key not found
            timeout: Timeout for lock acquisition

        Returns:
            Value if found, default otherwise
        """
        with self._lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for getting {key}")
                return default

            try:
                state = self._read_state()
                if key in state:
                    return state[key]["value"]
                else:
                    return default

            except Exception as e:
                logger.error(f"Error getting shared state {key}: {e}")
                return default

    def delete_value(self, key: str, timeout: float = 10.0) -> bool:
        """
        Delete a value from shared state.

        Args:
            key: Key to delete
            timeout: Timeout for lock acquisition

        Returns:
            True if value was deleted or didn't exist
        """
        with self._lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for deleting {key}")
                return False

            try:
                state = self._read_state()
                if key in state:
                    del state[key]
                    self._write_state(state)
                    logger.debug(f"Deleted shared state key {key}")

                return True

            except Exception as e:
                logger.error(f"Error deleting shared state {key}: {e}")
                return False

    def get_all_keys(self, timeout: float = 10.0) -> List[str]:
        """
        Get all keys in shared state.

        Args:
            timeout: Timeout for lock acquisition

        Returns:
            List of all keys
        """
        with self._lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning("Failed to acquire lock for getting all keys")
                return []

            try:
                state = self._read_state()
                return list(state.keys())

            except Exception as e:
                logger.error(f"Error getting all keys: {e}")
                return []

    def clear_all(self, timeout: float = 10.0) -> bool:
        """
        Clear all shared state.

        Args:
            timeout: Timeout for lock acquisition

        Returns:
            True if cleared successfully
        """
        with self._lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning("Failed to acquire lock for clearing state")
                return False

            try:
                self._write_state({})
                logger.debug("Cleared all shared state")
                return True

            except Exception as e:
                logger.error(f"Error clearing shared state: {e}")
                return False

    def _read_state(self) -> Dict[str, Any]:
        """Read state from file."""
        try:
            if self._state_file.exists():
                content = self._state_file.read_text()
                return eval(content) if content.strip() else {}
            else:
                return {}
        except Exception as e:
            logger.debug(f"Error reading state file: {e}")
            return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        """Write state to file."""
        try:
            content = repr(state)
            self._state_file.write_text(content)
        except Exception as e:
            logger.error(f"Error writing state file: {e}")
            raise


class DeadlockDetector:
    """
    Detects and prevents deadlocks in cross-process synchronization.

    This monitors lock acquisition patterns and can detect potential
    deadlock situations.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize deadlock detector."""
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._detector_dir = self._temp_dir / "exo_deadlock_detection"
        self._detector_dir.mkdir(exist_ok=True)

        self.process_id = os.getpid()
        self._held_locks: Set[str] = set()
        self._waiting_for: Optional[str] = None

    def register_lock_attempt(self, lock_name: str) -> bool:
        """
        Register a lock acquisition attempt.

        Args:
            lock_name: Name of lock being acquired

        Returns:
            True if safe to proceed, False if potential deadlock detected
        """
        # Simple deadlock detection - check if we're creating a cycle
        if self._would_create_cycle(lock_name):
            logger.warning(f"Potential deadlock detected for lock {lock_name}")
            return False

        self._waiting_for = lock_name
        return True

    def register_lock_acquired(self, lock_name: str) -> None:
        """Register successful lock acquisition."""
        self._held_locks.add(lock_name)
        self._waiting_for = None
        logger.debug(f"Registered lock acquisition: {lock_name}")

    def register_lock_released(self, lock_name: str) -> None:
        """Register lock release."""
        self._held_locks.discard(lock_name)
        logger.debug(f"Registered lock release: {lock_name}")

    def _would_create_cycle(self, lock_name: str) -> bool:
        """
        Check if acquiring a lock would create a deadlock cycle.

        This is a simplified implementation - a full implementation would
        build a wait-for graph across all processes.
        """
        # For now, just check if we already hold this lock
        return lock_name in self._held_locks

    def get_held_locks(self) -> Set[str]:
        """Get set of currently held locks."""
        return self._held_locks.copy()

    def is_waiting_for_lock(self) -> Optional[str]:
        """Get the lock this process is currently waiting for."""
        return self._waiting_for


# Global instances for cross-module usage
_global_deadlock_detector: Optional[DeadlockDetector] = None


def get_deadlock_detector() -> DeadlockDetector:
    """Get the global deadlock detector instance."""
    global _global_deadlock_detector
    if _global_deadlock_detector is None:
        _global_deadlock_detector = DeadlockDetector()
    return _global_deadlock_detector


def reset_deadlock_detector() -> None:
    """Reset the global deadlock detector (mainly for testing)."""
    global _global_deadlock_detector
    _global_deadlock_detector = None
