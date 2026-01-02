"""
Queue state synchronization for cross-process coordination.

This module provides shared queue state tracking, atomic queue closure operations,
and queue health monitoring to prevent race conditions during shutdown.
"""

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from loguru import logger

from exo.worker.runner.synchronization import CrossProcessLock, SharedStateManager, CrossProcessEvent, EventType


class QueueState(Enum):
    """States in the queue lifecycle."""
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"
    RECOVERING = "recovering"


class QueueHealthStatus(Enum):
    """Health status of a queue."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class QueueMetrics:
    """Metrics for queue monitoring."""
    queue_id: str
    state: QueueState
    health_status: QueueHealthStatus
    created_at: datetime
    last_activity: datetime
    put_operations: int = 0
    get_operations: int = 0
    failed_operations: int = 0
    current_size: int = 0
    max_size: int = 0
    average_wait_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class QueueRecoveryResult:
    """Result of a queue recovery operation."""
    success: bool
    recovered_items: int = 0
    recovery_time: float = 0.0
    error: Optional[Exception] = None


class SharedQueueState:
    """
    Manages shared state for a single queue across processes.
    
    This class provides atomic operations for updating queue state and
    coordinating between multiple processes that access the same queue.
    """
    
    def __init__(self, queue_id: str, max_size: int = 0):
        """
        Initialize shared queue state.
        
        Args:
            queue_id: Unique identifier for the queue
            max_size: Maximum queue size (0 for unlimited)
        """
        self.queue_id = queue_id
        self.max_size = max_size
        
        # Cross-process synchronization
        self._state_manager = SharedStateManager(f"queue_state_{queue_id}")
        self._lock = CrossProcessLock(f"queue_lock_{queue_id}")
        self._event_manager = CrossProcessEvent(f"queue_events_{queue_id}")
        
        # Initialize state if not exists
        self._initialize_state()
        
        logger.debug(f"Initialized shared state for queue {queue_id}")
    
    def _initialize_state(self) -> None:
        """Initialize the shared state if it doesn't exist."""
        with self._lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for initializing queue {self.queue_id}")
                return
            
            # Check if state already exists
            existing_state = self._state_manager.get_value("metrics")
            if existing_state is None:
                # Create initial metrics
                initial_metrics = QueueMetrics(
                    queue_id=self.queue_id,
                    state=QueueState.ACTIVE,
                    health_status=QueueHealthStatus.HEALTHY,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    max_size=self.max_size
                )
                
                self._state_manager.set_value("metrics", self._metrics_to_dict(initial_metrics))
                self._state_manager.set_value("process_count", 0)
                self._state_manager.set_value("active_operations", {})
                
                logger.debug(f"Initialized state for queue {self.queue_id}")
    
    def register_process(self, process_id: int) -> bool:
        """
        Register a process as using this queue.
        
        Args:
            process_id: Process ID to register
            
        Returns:
            True if registered successfully
        """
        with self._lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for registering process {process_id}")
                return False
            
            try:
                process_count = self._state_manager.get_value("process_count", 0)
                active_processes = self._state_manager.get_value("active_processes", set())
                
                if process_id not in active_processes:
                    active_processes.add(process_id)
                    self._state_manager.set_value("active_processes", active_processes)
                    self._state_manager.set_value("process_count", len(active_processes))
                    
                    logger.debug(f"Registered process {process_id} for queue {self.queue_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error registering process {process_id}: {e}")
                return False
    
    def unregister_process(self, process_id: int) -> bool:
        """
        Unregister a process from using this queue.
        
        Args:
            process_id: Process ID to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self._lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for unregistering process {process_id}")
                return False
            
            try:
                active_processes = self._state_manager.get_value("active_processes", set())
                
                if process_id in active_processes:
                    active_processes.remove(process_id)
                    self._state_manager.set_value("active_processes", active_processes)
                    self._state_manager.set_value("process_count", len(active_processes))
                    
                    logger.debug(f"Unregistered process {process_id} from queue {self.queue_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering process {process_id}: {e}")
                return False
    
    def update_state(self, new_state: QueueState, process_id: Optional[int] = None) -> bool:
        """
        Atomically update the queue state.
        
        Args:
            new_state: New state to set
            process_id: Process ID making the change
            
        Returns:
            True if state updated successfully
        """
        with self._lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for updating state to {new_state}")
                return False
            
            try:
                metrics_dict = self._state_manager.get_value("metrics", {})
                if not metrics_dict:
                    logger.warning(f"No metrics found for queue {self.queue_id}")
                    return False
                
                # Update state and last activity
                metrics_dict["state"] = new_state.value
                metrics_dict["last_activity"] = datetime.now().isoformat()
                
                self._state_manager.set_value("metrics", metrics_dict)
                
                # Signal state change event
                self._event_manager.signal(
                    EventType.CUSTOM,
                    {"state_change": new_state.value, "process_id": process_id}
                )
                
                logger.debug(f"Updated queue {self.queue_id} state to {new_state}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating state: {e}")
                return False
    
    def get_state(self) -> Optional[QueueState]:
        """
        Get the current queue state.
        
        Returns:
            Current QueueState or None if error
        """
        try:
            metrics_dict = self._state_manager.get_value("metrics", {})
            if metrics_dict and "state" in metrics_dict:
                return QueueState(metrics_dict["state"])
            return None
        except Exception as e:
            logger.debug(f"Error getting state: {e}")
            return None
    
    def update_metrics(self, **kwargs) -> bool:
        """
        Update queue metrics atomically.
        
        Args:
            **kwargs: Metric fields to update
            
        Returns:
            True if updated successfully
        """
        with self._lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                logger.warning(f"Failed to acquire lock for updating metrics")
                return False
            
            try:
                metrics_dict = self._state_manager.get_value("metrics", {})
                if not metrics_dict:
                    return False
                
                # Update provided metrics
                for key, value in kwargs.items():
                    if key in metrics_dict:
                        if isinstance(value, datetime):
                            metrics_dict[key] = value.isoformat()
                        else:
                            metrics_dict[key] = value
                
                # Always update last activity
                metrics_dict["last_activity"] = datetime.now().isoformat()
                
                self._state_manager.set_value("metrics", metrics_dict)
                return True
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return False
    
    def get_metrics(self) -> Optional[QueueMetrics]:
        """
        Get current queue metrics.
        
        Returns:
            QueueMetrics object or None if error
        """
        try:
            metrics_dict = self._state_manager.get_value("metrics", {})
            if not metrics_dict:
                return None
            
            return self._dict_to_metrics(metrics_dict)
            
        except Exception as e:
            logger.debug(f"Error getting metrics: {e}")
            return None
    
    def wait_for_state(self, target_state: QueueState, timeout: float = 10.0) -> bool:
        """
        Wait for the queue to reach a specific state.
        
        Args:
            target_state: State to wait for
            timeout: Maximum time to wait
            
        Returns:
            True if target state reached, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_state = self.get_state()
            if current_state == target_state:
                return True
            
            time.sleep(0.1)
        
        return False
    
    def is_safe_to_close(self) -> bool:
        """
        Check if it's safe to close the queue.
        
        Returns:
            True if safe to close (no active operations)
        """
        try:
            active_operations = self._state_manager.get_value("active_operations", {})
            process_count = self._state_manager.get_value("process_count", 0)
            
            # Safe to close if no active operations and only one process
            return len(active_operations) == 0 and process_count <= 1
            
        except Exception as e:
            logger.debug(f"Error checking if safe to close: {e}")
            return False
    
    def _metrics_to_dict(self, metrics: QueueMetrics) -> Dict[str, Any]:
        """Convert QueueMetrics to dictionary for storage."""
        return {
            "queue_id": metrics.queue_id,
            "state": metrics.state.value,
            "health_status": metrics.health_status.value,
            "created_at": metrics.created_at.isoformat(),
            "last_activity": metrics.last_activity.isoformat(),
            "put_operations": metrics.put_operations,
            "get_operations": metrics.get_operations,
            "failed_operations": metrics.failed_operations,
            "current_size": metrics.current_size,
            "max_size": metrics.max_size,
            "average_wait_time": metrics.average_wait_time,
            "error_count": metrics.error_count,
            "last_error": metrics.last_error
        }
    
    def _dict_to_metrics(self, metrics_dict: Dict[str, Any]) -> QueueMetrics:
        """Convert dictionary to QueueMetrics object."""
        return QueueMetrics(
            queue_id=metrics_dict.get("queue_id", self.queue_id),
            state=QueueState(metrics_dict.get("state", "active")),
            health_status=QueueHealthStatus(metrics_dict.get("health_status", "healthy")),
            created_at=datetime.fromisoformat(metrics_dict.get("created_at", datetime.now().isoformat())),
            last_activity=datetime.fromisoformat(metrics_dict.get("last_activity", datetime.now().isoformat())),
            put_operations=metrics_dict.get("put_operations", 0),
            get_operations=metrics_dict.get("get_operations", 0),
            failed_operations=metrics_dict.get("failed_operations", 0),
            current_size=metrics_dict.get("current_size", 0),
            max_size=metrics_dict.get("max_size", 0),
            average_wait_time=metrics_dict.get("average_wait_time", 0.0),
            error_count=metrics_dict.get("error_count", 0),
            last_error=metrics_dict.get("last_error")
        )


class QueueStateManager:
    """
    Manages shared state for multiple queues across processes.
    
    This class provides centralized queue state management with health monitoring,
    atomic closure operations, and recovery mechanisms.
    """
    
    def __init__(self):
        """Initialize the queue state manager."""
        self._queue_states: Dict[str, SharedQueueState] = {}
        self._lock = asyncio.Lock()
        
        # Global state management
        self._global_state = SharedStateManager("global_queue_state")
        self._global_lock = CrossProcessLock("global_queue_manager")
        
        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.debug("Initialized QueueStateManager")
    
    async def register_queue(self, queue_id: str, max_size: int = 0) -> SharedQueueState:
        """
        Register a queue for state management.
        
        Args:
            queue_id: Unique identifier for the queue
            max_size: Maximum queue size
            
        Returns:
            SharedQueueState instance for the queue
        """
        async with self._lock:
            if queue_id in self._queue_states:
                logger.debug(f"Queue {queue_id} already registered")
                return self._queue_states[queue_id]
            
            # Create shared state for the queue
            queue_state = SharedQueueState(queue_id, max_size)
            self._queue_states[queue_id] = queue_state
            
            # Register with global state
            await self._update_global_registry(queue_id, "register")
            
            logger.info(f"Registered queue {queue_id} for state management")
            return queue_state
    
    async def unregister_queue(self, queue_id: str) -> bool:
        """
        Unregister a queue from state management.
        
        Args:
            queue_id: Queue ID to unregister
            
        Returns:
            True if unregistered successfully
        """
        async with self._lock:
            if queue_id not in self._queue_states:
                logger.warning(f"Queue {queue_id} not found for unregistration")
                return False
            
            # Remove from tracking
            del self._queue_states[queue_id]
            
            # Update global registry
            await self._update_global_registry(queue_id, "unregister")
            
            logger.info(f"Unregistered queue {queue_id}")
            return True
    
    def get_queue_state(self, queue_id: str) -> Optional[SharedQueueState]:
        """
        Get the shared state for a queue.
        
        Args:
            queue_id: Queue ID to get state for
            
        Returns:
            SharedQueueState if found, None otherwise
        """
        return self._queue_states.get(queue_id)
    
    async def close_queue_atomically(self, queue_id: str, drain_timeout: float = 5.0) -> bool:
        """
        Atomically close a queue with proper coordination.
        
        Args:
            queue_id: Queue ID to close
            drain_timeout: Timeout for draining operations
            
        Returns:
            True if closed successfully
        """
        queue_state = self._queue_states.get(queue_id)
        if not queue_state:
            logger.warning(f"Queue {queue_id} not found for closing")
            return False
        
        logger.info(f"Starting atomic closure of queue {queue_id}")
        
        try:
            # Phase 1: Signal closure intent
            if not queue_state.update_state(QueueState.DRAINING):
                logger.error(f"Failed to signal draining for queue {queue_id}")
                return False
            
            # Phase 2: Wait for safe closure conditions
            start_time = time.time()
            while time.time() - start_time < drain_timeout:
                if queue_state.is_safe_to_close():
                    break
                await asyncio.sleep(0.1)
            
            # Phase 3: Mark as closing
            if not queue_state.update_state(QueueState.CLOSING):
                logger.error(f"Failed to signal closing for queue {queue_id}")
                return False
            
            # Phase 4: Final closure
            success = queue_state.update_state(QueueState.CLOSED)
            
            if success:
                logger.info(f"Successfully closed queue {queue_id}")
            else:
                logger.error(f"Failed to close queue {queue_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during atomic closure of queue {queue_id}: {e}")
            queue_state.update_state(QueueState.ERROR)
            return False
    
    async def recover_queue(self, queue_id: str) -> QueueRecoveryResult:
        """
        Attempt to recover a queue from error state.
        
        Args:
            queue_id: Queue ID to recover
            
        Returns:
            QueueRecoveryResult with recovery details
        """
        queue_state = self._queue_states.get(queue_id)
        if not queue_state:
            return QueueRecoveryResult(
                success=False,
                error=ValueError(f"Queue {queue_id} not found")
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting recovery for queue {queue_id}")
            
            # Mark as recovering
            queue_state.update_state(QueueState.RECOVERING)
            
            # Reset error counters
            queue_state.update_metrics(
                error_count=0,
                last_error=None,
                failed_operations=0
            )
            
            # Attempt to restore to active state
            if queue_state.update_state(QueueState.ACTIVE):
                recovery_time = time.time() - start_time
                
                logger.info(f"Successfully recovered queue {queue_id} in {recovery_time:.3f}s")
                
                return QueueRecoveryResult(
                    success=True,
                    recovery_time=recovery_time
                )
            else:
                raise RuntimeError("Failed to restore active state")
                
        except Exception as e:
            recovery_time = time.time() - start_time
            logger.error(f"Failed to recover queue {queue_id}: {e}")
            
            return QueueRecoveryResult(
                success=False,
                recovery_time=recovery_time,
                error=e
            )
    
    async def health_check_all_queues(self) -> Dict[str, QueueHealthStatus]:
        """
        Perform health check on all registered queues.
        
        Returns:
            Dictionary mapping queue IDs to health status
        """
        health_results = {}
        
        for queue_id, queue_state in self._queue_states.items():
            try:
                metrics = queue_state.get_metrics()
                if not metrics:
                    health_results[queue_id] = QueueHealthStatus.UNKNOWN
                    continue
                
                # Determine health based on various factors
                now = datetime.now()
                time_since_activity = (now - metrics.last_activity).total_seconds()
                
                if metrics.state == QueueState.ERROR:
                    health_status = QueueHealthStatus.UNHEALTHY
                elif metrics.error_count > 10 or time_since_activity > 300:  # 5 minutes
                    health_status = QueueHealthStatus.DEGRADED
                elif metrics.state == QueueState.ACTIVE and metrics.error_count == 0:
                    health_status = QueueHealthStatus.HEALTHY
                else:
                    health_status = QueueHealthStatus.DEGRADED
                
                # Update health status in metrics
                queue_state.update_metrics(health_status=health_status.value)
                health_results[queue_id] = health_status
                
            except Exception as e:
                logger.error(f"Error checking health for queue {queue_id}: {e}")
                health_results[queue_id] = QueueHealthStatus.UNKNOWN
        
        return health_results
    
    async def start_health_monitoring(self) -> None:
        """Start periodic health monitoring of all queues."""
        if self._health_check_task and not self._health_check_task.done():
            logger.debug("Health monitoring already running")
            return
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started queue health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped queue health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Main loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                health_results = await self.health_check_all_queues()
                
                # Log unhealthy queues
                unhealthy_queues = [
                    queue_id for queue_id, status in health_results.items()
                    if status == QueueHealthStatus.UNHEALTHY
                ]
                
                if unhealthy_queues:
                    logger.warning(f"Unhealthy queues detected: {unhealthy_queues}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _update_global_registry(self, queue_id: str, action: str) -> None:
        """Update the global queue registry."""
        try:
            with self._global_lock.acquire_context(timeout=5.0) as acquired:
                if not acquired:
                    logger.warning("Failed to acquire global lock for registry update")
                    return
                
                registry = self._global_state.get_value("queue_registry", set())
                
                if action == "register":
                    registry.add(queue_id)
                elif action == "unregister":
                    registry.discard(queue_id)
                
                self._global_state.set_value("queue_registry", registry)
                self._global_state.set_value("last_updated", datetime.now().isoformat())
                
        except Exception as e:
            logger.debug(f"Error updating global registry: {e}")
    
    def get_all_queue_metrics(self) -> Dict[str, Optional[QueueMetrics]]:
        """
        Get metrics for all registered queues.
        
        Returns:
            Dictionary mapping queue IDs to QueueMetrics
        """
        return {
            queue_id: queue_state.get_metrics()
            for queue_id, queue_state in self._queue_states.items()
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all queues.
        
        Returns:
            Dictionary with summary statistics
        """
        all_metrics = self.get_all_queue_metrics()
        
        total_queues = len(all_metrics)
        active_queues = sum(1 for m in all_metrics.values() if m and m.state == QueueState.ACTIVE)
        error_queues = sum(1 for m in all_metrics.values() if m and m.state == QueueState.ERROR)
        
        total_operations = sum(
            (m.put_operations + m.get_operations) for m in all_metrics.values() if m
        )
        total_errors = sum(m.error_count for m in all_metrics.values() if m)
        
        return {
            "total_queues": total_queues,
            "active_queues": active_queues,
            "error_queues": error_queues,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_operations, 1)
        }


# Global instance for cross-module usage
_global_queue_state_manager: Optional[QueueStateManager] = None


def get_queue_state_manager() -> QueueStateManager:
    """Get the global queue state manager instance."""
    global _global_queue_state_manager
    if _global_queue_state_manager is None:
        _global_queue_state_manager = QueueStateManager()
    return _global_queue_state_manager


def reset_queue_state_manager() -> None:
    """Reset the global queue state manager (mainly for testing)."""
    global _global_queue_state_manager
    _global_queue_state_manager = None