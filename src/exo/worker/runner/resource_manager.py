"""
Resource lifecycle management for preventing race conditions during shutdown.

This module provides dependency-aware resource cleanup with proper ordering and timeout handling
to ensure graceful termination without resource leaks or race conditions.
"""

import asyncio
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from loguru import logger


class ResourceState(Enum):
    """States in the resource lifecycle."""

    ACTIVE = "active"
    DRAINING = "draining"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class ResourceType(Enum):
    """Types of resources that can be managed."""

    QUEUE = "queue"
    CHANNEL = "channel"
    PROCESS = "process"
    FILE = "file"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class ResourceHandle:
    """Handle for tracking a managed resource."""

    id: str
    resource_type: ResourceType
    cleanup_order: int
    dependencies: Set[str] = field(default_factory=set)
    state: ResourceState = ResourceState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    cleanup_func: Optional[Callable[[], None]] = None
    async_cleanup_func: Optional[Callable[[], Any]] = None
    timeout: float = 10.0

    def __post_init__(self):
        """Validate the resource handle after initialization."""
        if self.cleanup_func is None and self.async_cleanup_func is None:
            raise ValueError(
                "Either cleanup_func or async_cleanup_func must be provided"
            )


@dataclass
class CleanupResult:
    """Result of a resource cleanup operation."""

    success: bool
    cleaned_resources: List[str] = field(default_factory=list)
    failed_resources: List[str] = field(default_factory=list)
    errors: List[Exception] = field(default_factory=list)
    total_time: float = 0.0


class ResourceManager:
    """
    Manages resource lifecycle with dependency-aware cleanup ordering.

    This class ensures resources are cleaned up in the correct order to prevent
    race conditions, with timeout-based fallbacks for robustness.
    """

    def __init__(self):
        """Initialize the resource manager."""
        self._resources: Dict[str, ResourceHandle] = {}
        self._resource_refs: Dict[
            str, Any
        ] = {}  # Keep weak references to actual resources
        self._lock = asyncio.Lock()
        self._cleanup_in_progress = False

    def register_resource(
        self,
        resource: Any,
        resource_type: ResourceType,
        cleanup_order: int,
        cleanup_func: Optional[Callable[[], None]] = None,
        async_cleanup_func: Optional[Callable[[], Any]] = None,
        dependencies: Optional[Set[str]] = None,
        timeout: float = 10.0,
        resource_id: Optional[str] = None,
    ) -> ResourceHandle:
        """
        Register a resource for lifecycle management.

        Args:
            resource: The actual resource object
            resource_type: Type of resource
            cleanup_order: Order for cleanup (lower numbers cleaned up first)
            cleanup_func: Synchronous cleanup function
            async_cleanup_func: Asynchronous cleanup function
            dependencies: Set of resource IDs this resource depends on
            timeout: Timeout for cleanup operations
            resource_id: Optional custom ID, otherwise generated

        Returns:
            ResourceHandle for the registered resource
        """
        if resource_id is None:
            resource_id = f"{resource_type.value}_{uuid4().hex[:8]}"

        if dependencies is None:
            dependencies = set()

        handle = ResourceHandle(
            id=resource_id,
            resource_type=resource_type,
            cleanup_order=cleanup_order,
            dependencies=dependencies,
            cleanup_func=cleanup_func,
            async_cleanup_func=async_cleanup_func,
            timeout=timeout,
        )

        self._resources[resource_id] = handle

        # Store weak reference to the actual resource to avoid memory leaks
        try:
            self._resource_refs[resource_id] = weakref.ref(resource)
        except TypeError:
            # Some objects can't be weakly referenced, store directly
            self._resource_refs[resource_id] = resource

        logger.debug(
            f"Registered resource {resource_id} (type: {resource_type}, order: {cleanup_order})"
        )
        return handle

    async def cleanup_resources(self, timeout: float = 30.0) -> CleanupResult:
        """
        Clean up all registered resources in dependency order.

        Args:
            timeout: Maximum time to wait for all cleanup operations

        Returns:
            CleanupResult with details of the cleanup operation
        """
        async with self._lock:
            if self._cleanup_in_progress:
                logger.warning("Cleanup already in progress")
                return CleanupResult(
                    success=False, errors=[RuntimeError("Cleanup already in progress")]
                )

            self._cleanup_in_progress = True

        start_time = datetime.now()
        result = CleanupResult(success=True)

        try:
            # Get cleanup order based on dependencies and cleanup_order
            cleanup_order = self._calculate_cleanup_order()

            logger.info(
                f"Starting cleanup of {len(cleanup_order)} resources with timeout {timeout}s"
            )

            # Clean up resources in calculated order
            for resource_id in cleanup_order:
                if resource_id not in self._resources:
                    continue

                handle = self._resources[resource_id]

                # Check if we've exceeded the overall timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    logger.warning(
                        "Overall cleanup timeout exceeded, skipping remaining resources"
                    )
                    result.success = False
                    break

                # Calculate remaining time for this resource
                remaining_timeout = min(handle.timeout, timeout - elapsed)

                try:
                    await self._cleanup_single_resource(handle, remaining_timeout)
                    result.cleaned_resources.append(resource_id)
                    logger.debug(f"Successfully cleaned up resource {resource_id}")

                except Exception as e:
                    logger.error(f"Failed to clean up resource {resource_id}: {e}")
                    result.failed_resources.append(resource_id)
                    result.errors.append(e)
                    result.success = False

                    # Continue with other resources even if one fails
                    handle.state = ResourceState.ERROR

            # Calculate total time
            result.total_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Cleanup completed in {result.total_time:.2f}s. "
                f"Success: {len(result.cleaned_resources)}, "
                f"Failed: {len(result.failed_resources)}"
            )

            return result

        finally:
            async with self._lock:
                self._cleanup_in_progress = False

    def _calculate_cleanup_order(self) -> List[str]:
        """
        Calculate the order in which resources should be cleaned up.

        This uses topological sorting to respect dependencies, with cleanup_order
        as a secondary sort key.

        Returns:
            List of resource IDs in cleanup order
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        in_degree: Dict[str, int] = {}

        for resource_id, handle in self._resources.items():
            if handle.state in (ResourceState.CLOSED, ResourceState.ERROR):
                continue

            graph[resource_id] = handle.dependencies.copy()
            in_degree[resource_id] = len(handle.dependencies)

        # Topological sort with cleanup_order as secondary key
        result = []
        queue = []

        # Find resources with no dependencies
        for resource_id in graph:
            if in_degree[resource_id] == 0:
                queue.append(resource_id)

        # Sort by cleanup_order (lower numbers first)
        queue.sort(key=lambda rid: self._resources[rid].cleanup_order)

        while queue:
            # Get next resource to process
            current = queue.pop(0)
            result.append(current)

            # Update dependencies
            for resource_id in graph:
                if current in graph[resource_id]:
                    graph[resource_id].remove(current)
                    in_degree[resource_id] -= 1

                    if in_degree[resource_id] == 0:
                        # Insert in sorted position based on cleanup_order
                        cleanup_order = self._resources[resource_id].cleanup_order
                        inserted = False
                        for i, queued_id in enumerate(queue):
                            if self._resources[queued_id].cleanup_order > cleanup_order:
                                queue.insert(i, resource_id)
                                inserted = True
                                break
                        if not inserted:
                            queue.append(resource_id)

        # Check for circular dependencies
        remaining = [rid for rid in graph if rid not in result]
        if remaining:
            logger.warning(f"Circular dependencies detected for resources: {remaining}")
            # Add remaining resources sorted by cleanup_order
            remaining.sort(key=lambda rid: self._resources[rid].cleanup_order)
            result.extend(remaining)

        return result

    async def _cleanup_single_resource(
        self, handle: ResourceHandle, timeout: float
    ) -> None:
        """
        Clean up a single resource with timeout handling.

        Args:
            handle: Resource handle to clean up
            timeout: Maximum time to wait for cleanup
        """
        handle.state = ResourceState.DRAINING

        try:
            if handle.async_cleanup_func:
                # Use async cleanup function
                await asyncio.wait_for(handle.async_cleanup_func(), timeout=timeout)
            elif handle.cleanup_func:
                # Use sync cleanup function in thread
                await asyncio.wait_for(
                    asyncio.to_thread(handle.cleanup_func), timeout=timeout
                )
            else:
                # Try to call common cleanup methods on the resource itself
                resource = self._get_resource(handle.id)
                if resource is not None:
                    await self._try_default_cleanup(resource, timeout)

            handle.state = ResourceState.CLOSED

        except asyncio.TimeoutError:
            logger.warning(f"Cleanup timeout for resource {handle.id}")
            handle.state = ResourceState.ERROR
            raise
        except Exception as e:
            logger.error(f"Cleanup error for resource {handle.id}: {e}")
            handle.state = ResourceState.ERROR
            raise

    def _get_resource(self, resource_id: str) -> Optional[Any]:
        """Get the actual resource object by ID."""
        ref = self._resource_refs.get(resource_id)
        if ref is None:
            return None

        # Handle weak references
        if isinstance(ref, weakref.ref):
            return ref()
        else:
            return ref

    async def _try_default_cleanup(self, resource: Any, timeout: float) -> None:
        """
        Try common cleanup methods on a resource.

        Args:
            resource: The resource object
            timeout: Timeout for cleanup operations
        """
        # Try common cleanup method names
        cleanup_methods = ["close", "cleanup", "shutdown", "stop", "terminate"]

        for method_name in cleanup_methods:
            if hasattr(resource, method_name):
                method = getattr(resource, method_name)
                if callable(method):
                    try:
                        if asyncio.iscoroutinefunction(method):
                            await asyncio.wait_for(method(), timeout=timeout)
                        else:
                            await asyncio.wait_for(
                                asyncio.to_thread(method), timeout=timeout
                            )
                        logger.debug(f"Successfully called {method_name}() on resource")
                        return
                    except Exception as e:
                        logger.debug(f"Failed to call {method_name}(): {e}")
                        continue

        logger.warning("No suitable cleanup method found for resource")

    def is_resource_active(self, handle: ResourceHandle) -> bool:
        """
        Check if a resource is currently active.

        Args:
            handle: Resource handle to check

        Returns:
            True if resource is active, False otherwise
        """
        return handle.state == ResourceState.ACTIVE

    def get_resource_dependencies(self, handle: ResourceHandle) -> List[ResourceHandle]:
        """
        Get the dependency handles for a resource.

        Args:
            handle: Resource handle to get dependencies for

        Returns:
            List of ResourceHandle objects this resource depends on
        """
        dependencies = []
        for dep_id in handle.dependencies:
            if dep_id in self._resources:
                dependencies.append(self._resources[dep_id])
        return dependencies

    def get_resource_state(self, resource_id: str) -> Optional[ResourceState]:
        """
        Get the current state of a resource.

        Args:
            resource_id: ID of the resource to check

        Returns:
            ResourceState if resource exists, None otherwise
        """
        handle = self._resources.get(resource_id)
        return handle.state if handle else None

    def get_all_resources(self) -> Dict[str, ResourceHandle]:
        """
        Get all registered resource handles.

        Returns:
            Dictionary mapping resource IDs to ResourceHandle objects
        """
        return self._resources.copy()

    def unregister_resource(self, resource_id: str) -> bool:
        """
        Unregister a resource from management.

        Args:
            resource_id: ID of resource to unregister

        Returns:
            True if resource was unregistered, False if not found
        """
        if resource_id in self._resources:
            del self._resources[resource_id]
            self._resource_refs.pop(resource_id, None)
            logger.debug(f"Unregistered resource {resource_id}")
            return True
        return False

    async def cleanup_single_resource_by_id(
        self, resource_id: str, timeout: float = 10.0
    ) -> bool:
        """
        Clean up a single resource by ID.

        Args:
            resource_id: ID of resource to clean up
            timeout: Timeout for cleanup operation

        Returns:
            True if cleanup succeeded, False otherwise
        """
        if resource_id not in self._resources:
            logger.warning(f"Resource {resource_id} not found for cleanup")
            return False

        handle = self._resources[resource_id]

        try:
            await self._cleanup_single_resource(handle, timeout)
            logger.debug(f"Successfully cleaned up resource {resource_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clean up resource {resource_id}: {e}")
            return False

    def get_resource_count_by_state(self) -> Dict[ResourceState, int]:
        """
        Get count of resources by state.

        Returns:
            Dictionary mapping ResourceState to count
        """
        counts = {state: 0 for state in ResourceState}
        for handle in self._resources.values():
            counts[handle.state] += 1
        return counts

    async def wait_for_state(
        self, resource_id: str, target_state: ResourceState, timeout: float = 10.0
    ) -> bool:
        """
        Wait for a resource to reach a specific state.

        Args:
            resource_id: ID of resource to wait for
            target_state: State to wait for
            timeout: Maximum time to wait

        Returns:
            True if resource reached target state, False if timeout or error
        """
        start_time = datetime.now()
        timeout_at = start_time + timedelta(seconds=timeout)

        while datetime.now() < timeout_at:
            if resource_id not in self._resources:
                return False

            if self._resources[resource_id].state == target_state:
                return True

            await asyncio.sleep(0.1)

        return False


# Global instance for cross-module usage
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


def reset_resource_manager() -> None:
    """Reset the global resource manager (mainly for testing)."""
    global _global_resource_manager
    _global_resource_manager = None
