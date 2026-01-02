"""
Comprehensive lifecycle event logging for multinode race condition debugging.

This module provides structured logging for all runner lifecycle events with timing analysis,
correlation IDs, and multi-node debugging support.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from loguru import logger


class LifecycleEventType(Enum):
    """Types of lifecycle events to track."""

    # Runner lifecycle
    RUNNER_CREATED = "runner_created"
    RUNNER_STARTING = "runner_starting"
    RUNNER_STARTED = "runner_started"
    RUNNER_STOPPING = "runner_stopping"
    RUNNER_STOPPED = "runner_stopped"
    RUNNER_FAILED = "runner_failed"
    RUNNER_RECOVERED = "runner_recovered"

    # Resource lifecycle
    RESOURCE_REGISTERED = "resource_registered"
    RESOURCE_CLEANUP_START = "resource_cleanup_start"
    RESOURCE_CLEANUP_COMPLETE = "resource_cleanup_complete"
    RESOURCE_CLEANUP_FAILED = "resource_cleanup_failed"

    # Shutdown coordination
    SHUTDOWN_INITIATED = "shutdown_initiated"
    SHUTDOWN_PHASE_STARTED = "shutdown_phase_started"
    SHUTDOWN_PHASE_COMPLETED = "shutdown_phase_completed"
    SHUTDOWN_COMPLETED = "shutdown_completed"
    SHUTDOWN_FAILED = "shutdown_failed"

    # Channel operations
    CHANNEL_CREATED = "channel_created"
    CHANNEL_DRAINING = "channel_draining"
    CHANNEL_DRAINED = "channel_drained"
    CHANNEL_CLOSED = "channel_closed"
    CHANNEL_ERROR = "channel_error"

    # Task operations
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_TIMEOUT = "task_timeout"

    # Health monitoring
    HEALTH_CHECK_STARTED = "health_check_started"
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_RECOVERY_STARTED = "health_recovery_started"
    HEALTH_RECOVERY_COMPLETED = "health_recovery_completed"

    # Error handling
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERY_STARTED = "error_recovery_started"
    ERROR_RECOVERY_COMPLETED = "error_recovery_completed"
    ERROR_ESCALATED = "error_escalated"


@dataclass
class LifecycleEvent:
    """Structured lifecycle event with timing and correlation information."""

    event_type: LifecycleEventType
    timestamp: datetime
    runner_id: str
    correlation_id: str
    component: str
    operation: str
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TimingContext:
    """Context for tracking operation timing."""

    operation: str
    start_time: float
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LifecycleLogger:
    """
    Comprehensive lifecycle event logger with timing analysis and correlation tracking.

    Features:
    - Structured logging with correlation IDs
    - Timing analysis for performance monitoring
    - Multi-node log aggregation support
    - Race condition debugging utilities
    """

    def __init__(
        self, log_dir: Optional[Path] = None, enable_file_logging: bool = True
    ):
        """
        Initialize the lifecycle logger.

        Args:
            log_dir: Directory for log files. If None, uses current directory.
            enable_file_logging: Whether to write structured logs to files.
        """
        self._log_dir = log_dir or Path.cwd() / "logs"
        self._enable_file_logging = enable_file_logging
        self._correlation_stack: List[str] = []
        self._timing_contexts: Dict[str, TimingContext] = {}
        self._event_buffer: List[LifecycleEvent] = []
        self._buffer_lock = asyncio.Lock()

        # Performance metrics
        self._event_counts: Dict[LifecycleEventType, int] = {}
        self._timing_stats: Dict[str, List[float]] = {}

        # Multi-node coordination
        self._node_id = str(uuid4())[:8]
        self._process_id = str(uuid4())[:8]

        if self._enable_file_logging:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Set up structured file logging."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Create structured log file
        log_file = self._log_dir / f"lifecycle_{self._node_id}_{self._process_id}.jsonl"

        # Add structured logger
        logger.add(
            log_file,
            format="{message}",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
            serialize=True,
            enqueue=True,
        )

        logger.info(
            f"Lifecycle logging initialized: node_id={self._node_id}, process_id={self._process_id}"
        )

    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID for tracking related operations."""
        correlation_id = str(uuid4())[:12]
        logger.debug(f"Generated correlation ID: {correlation_id}")
        return correlation_id

    def push_correlation_id(self, correlation_id: str):
        """Push a correlation ID onto the stack for nested operations."""
        self._correlation_stack.append(correlation_id)
        logger.debug(
            f"Pushed correlation ID: {correlation_id}, stack depth: {len(self._correlation_stack)}"
        )

    def pop_correlation_id(self) -> Optional[str]:
        """Pop the current correlation ID from the stack."""
        if self._correlation_stack:
            correlation_id = self._correlation_stack.pop()
            logger.debug(
                f"Popped correlation ID: {correlation_id}, stack depth: {len(self._correlation_stack)}"
            )
            return correlation_id
        return None

    def get_current_correlation_id(self) -> str:
        """Get the current correlation ID or generate a new one."""
        if self._correlation_stack:
            return self._correlation_stack[-1]
        return self.generate_correlation_id()

    @asynccontextmanager
    async def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID tracking."""
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()

        self.push_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            self.pop_correlation_id()

    @asynccontextmanager
    async def timing_context(
        self,
        operation: str,
        runner_id: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for timing operations."""
        correlation_id = self.get_current_correlation_id()
        timing_key = f"{correlation_id}_{operation}"

        start_time = time.perf_counter()
        timing_ctx = TimingContext(
            operation=operation,
            start_time=start_time,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        self._timing_contexts[timing_key] = timing_ctx

        # Log operation start
        await self.log_event(
            event_type=LifecycleEventType.RUNNER_STARTING,  # Generic start event
            runner_id=runner_id,
            component=component,
            operation=operation,
            correlation_id=correlation_id,
            metadata=timing_ctx.metadata,
        )

        try:
            yield correlation_id
        except Exception as e:
            # Log operation failure
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self.log_event(
                event_type=LifecycleEventType.ERROR_OCCURRED,
                runner_id=runner_id,
                component=component,
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                metadata=timing_ctx.metadata,
            )
            raise
        finally:
            # Log operation completion
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self.log_event(
                event_type=LifecycleEventType.RUNNER_STARTED,  # Generic completion event
                runner_id=runner_id,
                component=component,
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=duration_ms,
                metadata=timing_ctx.metadata,
            )

            # Update timing statistics
            if operation not in self._timing_stats:
                self._timing_stats[operation] = []
            self._timing_stats[operation].append(duration_ms)

            # Clean up timing context
            self._timing_contexts.pop(timing_key, None)

    async def log_event(
        self,
        event_type: LifecycleEventType,
        runner_id: str,
        component: str,
        operation: str,
        correlation_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a lifecycle event with structured information.

        Args:
            event_type: Type of lifecycle event
            runner_id: ID of the runner involved
            component: Component generating the event
            operation: Operation being performed
            correlation_id: Correlation ID for tracking related events
            duration_ms: Duration of the operation in milliseconds
            success: Whether the operation was successful
            error_message: Error message if operation failed
            metadata: Additional metadata for the event
        """
        if correlation_id is None:
            correlation_id = self.get_current_correlation_id()

        event = LifecycleEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            runner_id=runner_id,
            correlation_id=correlation_id,
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Add node and process information
        event.metadata.update(
            {"node_id": self._node_id, "process_id": self._process_id}
        )

        # Buffer the event
        async with self._buffer_lock:
            self._event_buffer.append(event)

            # Update event counts
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        # Log to structured logger
        log_data = event.to_dict()

        # Choose log level based on event type and success
        if not success or event_type in (
            LifecycleEventType.ERROR_OCCURRED,
            LifecycleEventType.RUNNER_FAILED,
            LifecycleEventType.SHUTDOWN_FAILED,
        ):
            logger.error(f"LIFECYCLE_EVENT: {json.dumps(log_data)}")
        elif event_type in (
            LifecycleEventType.RUNNER_RECOVERED,
            LifecycleEventType.SHUTDOWN_COMPLETED,
            LifecycleEventType.HEALTH_RECOVERY_COMPLETED,
        ):
            logger.success(f"LIFECYCLE_EVENT: {json.dumps(log_data)}")
        else:
            logger.info(f"LIFECYCLE_EVENT: {json.dumps(log_data)}")

    async def log_runner_created(
        self, runner_id: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Log runner creation event."""
        await self.log_event(
            event_type=LifecycleEventType.RUNNER_CREATED,
            runner_id=runner_id,
            component="RunnerSupervisor",
            operation="create_runner",
            metadata=metadata,
        )

    async def log_runner_starting(
        self, runner_id: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Log runner starting event."""
        await self.log_event(
            event_type=LifecycleEventType.RUNNER_STARTING,
            runner_id=runner_id,
            component="RunnerSupervisor",
            operation="start_runner",
            metadata=metadata,
        )

    async def log_runner_started(
        self,
        runner_id: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log runner started event."""
        await self.log_event(
            event_type=LifecycleEventType.RUNNER_STARTED,
            runner_id=runner_id,
            component="RunnerSupervisor",
            operation="start_runner",
            duration_ms=duration_ms,
            metadata=metadata,
        )

    async def log_shutdown_initiated(
        self, runner_id: str, timeout: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """Log shutdown initiation event."""
        shutdown_metadata = {"timeout": timeout}
        if metadata:
            shutdown_metadata.update(metadata)

        await self.log_event(
            event_type=LifecycleEventType.SHUTDOWN_INITIATED,
            runner_id=runner_id,
            component="ShutdownCoordinator",
            operation="initiate_shutdown",
            metadata=shutdown_metadata,
        )

    async def log_shutdown_phase(
        self,
        runner_id: str,
        phase: str,
        duration_ms: Optional[float] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log shutdown phase event."""
        phase_metadata = {"phase": phase}
        if metadata:
            phase_metadata.update(metadata)

        event_type = (
            LifecycleEventType.SHUTDOWN_PHASE_COMPLETED
            if success
            else LifecycleEventType.SHUTDOWN_FAILED
        )

        await self.log_event(
            event_type=event_type,
            runner_id=runner_id,
            component="ShutdownCoordinator",
            operation=f"shutdown_phase_{phase}",
            duration_ms=duration_ms,
            success=success,
            metadata=phase_metadata,
        )

    async def log_resource_operation(
        self,
        runner_id: str,
        resource_id: str,
        operation: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log resource operation event."""
        resource_metadata = {"resource_id": resource_id}
        if metadata:
            resource_metadata.update(metadata)

        # Map operation to event type
        event_type_map = {
            "register": LifecycleEventType.RESOURCE_REGISTERED,
            "cleanup_start": LifecycleEventType.RESOURCE_CLEANUP_START,
            "cleanup_complete": LifecycleEventType.RESOURCE_CLEANUP_COMPLETE,
            "cleanup_failed": LifecycleEventType.RESOURCE_CLEANUP_FAILED,
        }

        event_type = event_type_map.get(
            operation, LifecycleEventType.RESOURCE_REGISTERED
        )

        await self.log_event(
            event_type=event_type,
            runner_id=runner_id,
            component="ResourceManager",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=resource_metadata,
        )

    async def log_health_check(
        self,
        runner_id: str,
        success: bool,
        duration_ms: float,
        health_score: float,
        issues: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log health check event."""
        health_metadata = {
            "health_score": health_score,
            "issues": issues,
            "issue_count": len(issues),
        }
        if metadata:
            health_metadata.update(metadata)

        event_type = (
            LifecycleEventType.HEALTH_CHECK_PASSED
            if success
            else LifecycleEventType.HEALTH_CHECK_FAILED
        )

        await self.log_event(
            event_type=event_type,
            runner_id=runner_id,
            component="RunnerSupervisor",
            operation="health_check",
            duration_ms=duration_ms,
            success=success,
            metadata=health_metadata,
        )

    async def log_error_handling(
        self,
        runner_id: str,
        error: Exception,
        component: str,
        operation: str,
        recovery_action: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log error handling event."""
        error_metadata = {
            "error_type": type(error).__name__,
            "recovery_action": recovery_action,
        }
        if metadata:
            error_metadata.update(metadata)

        event_type = (
            LifecycleEventType.ERROR_RECOVERY_COMPLETED
            if success
            else LifecycleEventType.ERROR_ESCALATED
        )

        await self.log_event(
            event_type=event_type,
            runner_id=runner_id,
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_message=str(error),
            metadata=error_metadata,
        )

    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        stats = {}

        for operation, timings in self._timing_stats.items():
            if timings:
                stats[operation] = {
                    "count": len(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "avg_ms": sum(timings) / len(timings),
                    "total_ms": sum(timings),
                }

                # Calculate percentiles
                sorted_timings = sorted(timings)
                n = len(sorted_timings)
                stats[operation].update(
                    {
                        "p50_ms": sorted_timings[n // 2],
                        "p90_ms": sorted_timings[int(n * 0.9)],
                        "p95_ms": sorted_timings[int(n * 0.95)],
                        "p99_ms": sorted_timings[int(n * 0.99)],
                    }
                )

        return stats

    def get_event_counts(self) -> Dict[str, int]:
        """Get event counts by type."""
        return {
            event_type.value: count for event_type, count in self._event_counts.items()
        }

    async def get_recent_events(
        self,
        hours: int = 1,
        event_types: Optional[Set[LifecycleEventType]] = None,
        runner_id: Optional[str] = None,
    ) -> List[LifecycleEvent]:
        """Get recent events matching the specified criteria."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        async with self._buffer_lock:
            filtered_events = []

            for event in self._event_buffer:
                # Time filter
                if event.timestamp < cutoff_time:
                    continue

                # Event type filter
                if event_types and event.event_type not in event_types:
                    continue

                # Runner ID filter
                if runner_id and event.runner_id != runner_id:
                    continue

                filtered_events.append(event)

        return sorted(filtered_events, key=lambda e: e.timestamp, reverse=True)

    async def export_events(
        self,
        output_file: Path,
        hours: Optional[int] = None,
        event_types: Optional[Set[LifecycleEventType]] = None,
        runner_id: Optional[str] = None,
    ):
        """Export events to a file for analysis."""
        if hours:
            events = await self.get_recent_events(hours, event_types, runner_id)
        else:
            async with self._buffer_lock:
                events = list(self._event_buffer)

        with open(output_file, "w") as f:
            for event in events:
                f.write(event.to_json() + "\n")

        logger.info(f"Exported {len(events)} events to {output_file}")

    async def cleanup_old_events(self, hours: int = 24):
        """Clean up old events from the buffer to prevent memory growth."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        async with self._buffer_lock:
            original_count = len(self._event_buffer)
            self._event_buffer = [
                e for e in self._event_buffer if e.timestamp >= cutoff_time
            ]
            cleaned_count = original_count - len(self._event_buffer)

            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old lifecycle events")


# Global instance for cross-module usage
_global_lifecycle_logger: Optional[LifecycleLogger] = None


def get_lifecycle_logger() -> LifecycleLogger:
    """Get the global lifecycle logger instance."""
    global _global_lifecycle_logger
    if _global_lifecycle_logger is None:
        _global_lifecycle_logger = LifecycleLogger()
    return _global_lifecycle_logger


def reset_lifecycle_logger() -> None:
    """Reset the global lifecycle logger (mainly for testing)."""
    global _global_lifecycle_logger
    _global_lifecycle_logger = None
