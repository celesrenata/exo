"""
Enhanced queue operations with state checking and dependency resolution.

This module provides safe queue operations with timeout handling, progress tracking,
and proper cleanup with dependency resolution to prevent race conditions.
"""

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

try:
    from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender, WouldBlock
except ImportError:
    # Fallback for missing imports
    class ClosedResourceError(Exception):
        pass

    class WouldBlock(Exception):
        pass

    MpReceiver = Any
    MpSender = Any

from exo.worker.runner.queue_state_manager import (
    QueueState,
    QueueStateManager,
    get_queue_state_manager,
)


class OperationType(Enum):
    """Types of queue operations."""

    PUT = "put"
    GET = "get"
    DRAIN = "drain"
    CLOSE = "close"


class OperationResult(Enum):
    """Results of queue operations."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    QUEUE_CLOSED = "queue_closed"
    QUEUE_FULL = "queue_full"
    QUEUE_EMPTY = "queue_empty"
    ERROR = "error"


@dataclass
class OperationContext:
    """Context for tracking queue operations."""

    operation_id: str
    operation_type: OperationType
    queue_id: str
    started_at: datetime
    timeout: float
    process_id: int = field(
        default_factory=lambda: mp.current_process().pid
        if hasattr(mp.current_process(), "pid")
        else 0
    )
    completed_at: Optional[datetime] = None
    result: Optional[OperationResult] = None
    error: Optional[Exception] = None


@dataclass
class DrainProgress:
    """Progress tracking for queue draining operations."""

    items_drained: int = 0
    total_estimated: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    completion_percentage: float = 0.0
    items_per_second: float = 0.0


@dataclass
class QueueDependency:
    """Represents a dependency between queues."""

    dependent_queue: str
    dependency_queue: str
    dependency_type: str  # "producer", "consumer", "bidirectional"
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Make QueueDependency hashable for use in sets."""
        return hash((self.dependent_queue, self.dependency_queue, self.dependency_type))


class SafeQueueOperations:
    """
    Provides safe queue operations with state checking and timeout handling.

    This class wraps multiprocessing queue operations to prevent race conditions
    and provide better error handling during shutdown scenarios.
    """

    def __init__(self, queue_state_manager: Optional[QueueStateManager] = None):
        """
        Initialize safe queue operations.

        Args:
            queue_state_manager: Optional QueueStateManager instance
        """
        self._queue_state_manager = queue_state_manager or get_queue_state_manager()
        self._active_operations: Dict[str, OperationContext] = {}
        self._operation_lock = asyncio.Lock()

        # Dependency tracking
        self._dependencies: Dict[str, Set[QueueDependency]] = {}
        self._dependency_lock = asyncio.Lock()

        logger.debug("Initialized SafeQueueOperations")

    async def safe_put(
        self,
        queue: Union[MpSender, mp.Queue],
        item: Any,
        queue_id: str,
        timeout: float = 5.0,
        check_state: bool = True,
    ) -> Tuple[OperationResult, Optional[Exception]]:
        """
        Safely put an item into a queue with state checking.

        Args:
            queue: Queue or sender to put item into
            item: Item to put
            queue_id: Identifier for the queue
            timeout: Maximum time to wait
            check_state: Whether to check queue state before operation

        Returns:
            Tuple of (OperationResult, Optional[Exception])
        """
        operation_id = f"put_{uuid4().hex[:8]}"
        context = OperationContext(
            operation_id=operation_id,
            operation_type=OperationType.PUT,
            queue_id=queue_id,
            started_at=datetime.now(),
            timeout=timeout,
        )

        async with self._operation_lock:
            self._active_operations[operation_id] = context

        try:
            # Check queue state if requested
            if check_state:
                queue_state = await self._get_queue_state(queue_id)
                if queue_state and queue_state not in (
                    QueueState.ACTIVE,
                    QueueState.DRAINING,
                ):
                    context.result = OperationResult.QUEUE_CLOSED
                    return OperationResult.QUEUE_CLOSED, None

            # Perform the put operation with timeout
            result, error = await self._perform_put_operation(queue, item, timeout)

            context.result = result
            context.error = error
            context.completed_at = datetime.now()

            # Update queue metrics
            await self._update_operation_metrics(queue_id, OperationType.PUT, result)

            return result, error

        finally:
            async with self._operation_lock:
                self._active_operations.pop(operation_id, None)

    async def safe_get(
        self,
        queue: Union[MpReceiver, mp.Queue],
        queue_id: str,
        timeout: float = 5.0,
        check_state: bool = True,
    ) -> Tuple[OperationResult, Any, Optional[Exception]]:
        """
        Safely get an item from a queue with state checking.

        Args:
            queue: Queue or receiver to get item from
            queue_id: Identifier for the queue
            timeout: Maximum time to wait
            check_state: Whether to check queue state before operation

        Returns:
            Tuple of (OperationResult, item, Optional[Exception])
        """
        operation_id = f"get_{uuid4().hex[:8]}"
        context = OperationContext(
            operation_id=operation_id,
            operation_type=OperationType.GET,
            queue_id=queue_id,
            started_at=datetime.now(),
            timeout=timeout,
        )

        async with self._operation_lock:
            self._active_operations[operation_id] = context

        try:
            # Check queue state if requested
            if check_state:
                queue_state = await self._get_queue_state(queue_id)
                if queue_state == QueueState.CLOSED:
                    context.result = OperationResult.QUEUE_CLOSED
                    return OperationResult.QUEUE_CLOSED, None, None

            # Perform the get operation with timeout
            result, item, error = await self._perform_get_operation(queue, timeout)

            context.result = result
            context.error = error
            context.completed_at = datetime.now()

            # Update queue metrics
            await self._update_operation_metrics(queue_id, OperationType.GET, result)

            return result, item, error

        finally:
            async with self._operation_lock:
                self._active_operations.pop(operation_id, None)

    async def drain_queue_with_progress(
        self,
        queue: Union[MpReceiver, mp.Queue],
        queue_id: str,
        timeout: float = 30.0,
        progress_callback: Optional[Callable[[DrainProgress], None]] = None,
    ) -> Tuple[OperationResult, List[Any], DrainProgress]:
        """
        Drain a queue with progress tracking.

        Args:
            queue: Queue or receiver to drain
            queue_id: Identifier for the queue
            timeout: Maximum time to spend draining
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (OperationResult, drained_items, DrainProgress)
        """
        operation_id = f"drain_{uuid4().hex[:8]}"
        context = OperationContext(
            operation_id=operation_id,
            operation_type=OperationType.DRAIN,
            queue_id=queue_id,
            started_at=datetime.now(),
            timeout=timeout,
        )

        progress = DrainProgress()
        drained_items = []

        async with self._operation_lock:
            self._active_operations[operation_id] = context

        try:
            logger.info(f"Starting drain of queue {queue_id} with timeout {timeout}s")

            start_time = time.time()
            last_progress_update = start_time

            while time.time() - start_time < timeout:
                try:
                    # Try to get an item without blocking
                    result, item, error = await self._perform_get_operation(
                        queue, timeout=0.1
                    )

                    if result == OperationResult.SUCCESS:
                        drained_items.append(item)
                        progress.items_drained += 1
                        progress.last_update = datetime.now()

                        # Update progress metrics
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            progress.items_per_second = progress.items_drained / elapsed

                        # Call progress callback if provided
                        if (
                            progress_callback
                            and time.time() - last_progress_update > 1.0
                        ):
                            progress_callback(progress)
                            last_progress_update = time.time()

                    elif result == OperationResult.QUEUE_EMPTY:
                        # No more items available
                        break
                    elif result == OperationResult.QUEUE_CLOSED:
                        # Queue was closed during draining
                        break
                    else:
                        # Other error, continue trying for a bit
                        await asyncio.sleep(0.01)

                except Exception as e:
                    logger.debug(f"Error during drain: {e}")
                    await asyncio.sleep(0.01)

            # Final progress update
            total_time = time.time() - start_time
            progress.completion_percentage = 100.0  # We drained what we could
            if total_time > 0:
                progress.items_per_second = progress.items_drained / total_time

            if progress_callback:
                progress_callback(progress)

            context.result = OperationResult.SUCCESS
            context.completed_at = datetime.now()

            logger.info(
                f"Drained {progress.items_drained} items from queue {queue_id} in {total_time:.3f}s"
            )

            return OperationResult.SUCCESS, drained_items, progress

        except Exception as e:
            logger.error(f"Error draining queue {queue_id}: {e}")
            context.result = OperationResult.ERROR
            context.error = e
            return OperationResult.ERROR, drained_items, progress

        finally:
            async with self._operation_lock:
                self._active_operations.pop(operation_id, None)

    async def cleanup_queue_with_dependencies(
        self, queue_id: str, cleanup_func: Callable[[], None], timeout: float = 10.0
    ) -> OperationResult:
        """
        Clean up a queue with proper dependency resolution.

        Args:
            queue_id: Queue to clean up
            cleanup_func: Function to perform the actual cleanup
            timeout: Maximum time for cleanup

        Returns:
            OperationResult indicating success or failure
        """
        operation_id = f"cleanup_{uuid4().hex[:8]}"
        context = OperationContext(
            operation_id=operation_id,
            operation_type=OperationType.CLOSE,
            queue_id=queue_id,
            started_at=datetime.now(),
            timeout=timeout,
        )

        async with self._operation_lock:
            self._active_operations[operation_id] = context

        try:
            logger.info(f"Starting cleanup of queue {queue_id}")

            # Check dependencies
            dependencies = await self._get_queue_dependencies(queue_id)

            # Wait for dependent queues to be ready for cleanup
            for dep in dependencies:
                if not await self._wait_for_dependency_ready(
                    dep, timeout / len(dependencies) if dependencies else timeout
                ):
                    logger.warning(
                        f"Dependency {dep.dependency_queue} not ready for cleanup"
                    )

            # Perform the cleanup
            start_time = time.time()

            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await asyncio.wait_for(cleanup_func(), timeout=timeout)
                else:
                    await asyncio.wait_for(
                        asyncio.to_thread(cleanup_func), timeout=timeout
                    )

                cleanup_time = time.time() - start_time
                logger.info(
                    f"Successfully cleaned up queue {queue_id} in {cleanup_time:.3f}s"
                )

                context.result = OperationResult.SUCCESS
                return OperationResult.SUCCESS

            except asyncio.TimeoutError:
                logger.warning(f"Cleanup timeout for queue {queue_id}")
                context.result = OperationResult.TIMEOUT
                return OperationResult.TIMEOUT

        except Exception as e:
            logger.error(f"Error cleaning up queue {queue_id}: {e}")
            context.result = OperationResult.ERROR
            context.error = e
            return OperationResult.ERROR

        finally:
            async with self._operation_lock:
                self._active_operations.pop(operation_id, None)

    async def add_queue_dependency(
        self,
        dependent_queue: str,
        dependency_queue: str,
        dependency_type: str = "consumer",
    ) -> None:
        """
        Add a dependency relationship between queues.

        Args:
            dependent_queue: Queue that depends on another
            dependency_queue: Queue that is depended upon
            dependency_type: Type of dependency
        """
        async with self._dependency_lock:
            if dependent_queue not in self._dependencies:
                self._dependencies[dependent_queue] = set()

            dependency = QueueDependency(
                dependent_queue=dependent_queue,
                dependency_queue=dependency_queue,
                dependency_type=dependency_type,
            )

            self._dependencies[dependent_queue].add(dependency)

            logger.debug(
                f"Added dependency: {dependent_queue} -> {dependency_queue} ({dependency_type})"
            )

    async def remove_queue_dependency(
        self, dependent_queue: str, dependency_queue: str
    ) -> None:
        """
        Remove a dependency relationship between queues.

        Args:
            dependent_queue: Queue that depends on another
            dependency_queue: Queue that is depended upon
        """
        async with self._dependency_lock:
            if dependent_queue in self._dependencies:
                # Remove matching dependencies
                self._dependencies[dependent_queue] = {
                    dep
                    for dep in self._dependencies[dependent_queue]
                    if dep.dependency_queue != dependency_queue
                }

                # Clean up empty sets
                if not self._dependencies[dependent_queue]:
                    del self._dependencies[dependent_queue]

                logger.debug(
                    f"Removed dependency: {dependent_queue} -> {dependency_queue}"
                )

    async def _perform_put_operation(
        self, queue: Union[MpSender, mp.Queue], item: Any, timeout: float
    ) -> Tuple[OperationResult, Optional[Exception]]:
        """Perform the actual put operation with timeout."""
        try:
            if hasattr(queue, "send_nowait"):
                # MpSender interface
                try:
                    queue.send_nowait(item)
                    return OperationResult.SUCCESS, None
                except WouldBlock:
                    # Try blocking send with timeout simulation
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            queue.send_nowait(item)
                            return OperationResult.SUCCESS, None
                        except WouldBlock:
                            await asyncio.sleep(0.01)
                    return OperationResult.TIMEOUT, None
                except ClosedResourceError as e:
                    return OperationResult.QUEUE_CLOSED, e
            else:
                # Standard mp.Queue interface
                try:
                    queue.put_nowait(item)
                    return OperationResult.SUCCESS, None
                except Exception:
                    # Try blocking put with timeout
                    await asyncio.wait_for(
                        asyncio.to_thread(queue.put, item, True, timeout),
                        timeout=timeout,
                    )
                    return OperationResult.SUCCESS, None

        except asyncio.TimeoutError:
            return OperationResult.TIMEOUT, None
        except Exception as e:
            if "closed" in str(e).lower():
                return OperationResult.QUEUE_CLOSED, e
            return OperationResult.ERROR, e

    async def _perform_get_operation(
        self, queue: Union[MpReceiver, mp.Queue], timeout: float
    ) -> Tuple[OperationResult, Any, Optional[Exception]]:
        """Perform the actual get operation with timeout."""
        try:
            if hasattr(queue, "receive_nowait"):
                # MpReceiver interface
                try:
                    item = queue.receive_nowait()
                    return OperationResult.SUCCESS, item, None
                except WouldBlock:
                    if timeout <= 0:
                        return OperationResult.QUEUE_EMPTY, None, None

                    # Try blocking receive with timeout simulation
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            item = queue.receive_nowait()
                            return OperationResult.SUCCESS, item, None
                        except WouldBlock:
                            await asyncio.sleep(0.01)
                    return OperationResult.TIMEOUT, None, None
                except ClosedResourceError as e:
                    return OperationResult.QUEUE_CLOSED, None, e
            else:
                # Standard mp.Queue interface
                try:
                    item = queue.get_nowait()
                    return OperationResult.SUCCESS, item, None
                except Exception:
                    if timeout <= 0:
                        return OperationResult.QUEUE_EMPTY, None, None

                    # Try blocking get with timeout
                    item = await asyncio.wait_for(
                        asyncio.to_thread(queue.get, True, timeout), timeout=timeout
                    )
                    return OperationResult.SUCCESS, item, None

        except asyncio.TimeoutError:
            return OperationResult.TIMEOUT, None, None
        except Exception as e:
            if "closed" in str(e).lower() or "empty" in str(e).lower():
                return OperationResult.QUEUE_EMPTY, None, e
            return OperationResult.ERROR, None, e

    async def _get_queue_state(self, queue_id: str) -> Optional[QueueState]:
        """Get the current state of a queue."""
        queue_state = self._queue_state_manager.get_queue_state(queue_id)
        if queue_state:
            return queue_state.get_state()
        return None

    async def _update_operation_metrics(
        self, queue_id: str, operation_type: OperationType, result: OperationResult
    ) -> None:
        """Update metrics for a queue operation."""
        try:
            queue_state = self._queue_state_manager.get_queue_state(queue_id)
            if not queue_state:
                return

            if operation_type == OperationType.PUT:
                if result == OperationResult.SUCCESS:
                    metrics = queue_state.get_metrics()
                    if metrics:
                        queue_state.update_metrics(
                            put_operations=metrics.put_operations + 1
                        )
                else:
                    metrics = queue_state.get_metrics()
                    if metrics:
                        queue_state.update_metrics(
                            failed_operations=metrics.failed_operations + 1
                        )
            elif operation_type == OperationType.GET:
                if result == OperationResult.SUCCESS:
                    metrics = queue_state.get_metrics()
                    if metrics:
                        queue_state.update_metrics(
                            get_operations=metrics.get_operations + 1
                        )
                else:
                    metrics = queue_state.get_metrics()
                    if metrics:
                        queue_state.update_metrics(
                            failed_operations=metrics.failed_operations + 1
                        )

        except Exception as e:
            if hasattr(logger, "debug"):
                logger.debug(f"Error updating operation metrics: {e}")
            else:
                print("Error updating operation metrics: {e}")

    async def _get_queue_dependencies(self, queue_id: str) -> List[QueueDependency]:
        """Get dependencies for a queue."""
        async with self._dependency_lock:
            return list(self._dependencies.get(queue_id, set()))

    async def _wait_for_dependency_ready(
        self, dependency: QueueDependency, timeout: float
    ) -> bool:
        """Wait for a dependency to be ready for cleanup."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            queue_state = await self._get_queue_state(dependency.dependency_queue)

            if queue_state in (QueueState.CLOSED, QueueState.DRAINING):
                return True

            await asyncio.sleep(0.1)

        return False

    def get_active_operations(self) -> Dict[str, OperationContext]:
        """Get all currently active operations."""
        return self._active_operations.copy()

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about queue operations."""
        operations = list(self._active_operations.values())

        return {
            "active_operations": len(operations),
            "operations_by_type": {
                op_type.value: sum(
                    1 for op in operations if op.operation_type == op_type
                )
                for op_type in OperationType
            },
            "average_operation_time": sum(
                (op.completed_at - op.started_at).total_seconds()
                for op in operations
                if op.completed_at
            )
            / max(len([op for op in operations if op.completed_at]), 1),
        }


# Global instance for cross-module usage
_global_safe_queue_operations: Optional[SafeQueueOperations] = None


def get_safe_queue_operations() -> SafeQueueOperations:
    """Get the global safe queue operations instance."""
    global _global_safe_queue_operations
    if _global_safe_queue_operations is None:
        _global_safe_queue_operations = SafeQueueOperations()
    return _global_safe_queue_operations


def reset_safe_queue_operations() -> None:
    """Reset the global safe queue operations (mainly for testing)."""
    global _global_safe_queue_operations
    _global_safe_queue_operations = None
