"""
Runner module for EXO distributed inference system.

This module contains the core runner infrastructure including shutdown coordination,
resource management, and synchronization primitives for preventing race conditions
in multiprocessing environments.
"""

from .resource_manager import (
    CleanupResult,
    ResourceHandle,
    ResourceManager,
    ResourceState,
    ResourceType,
    get_resource_manager,
    reset_resource_manager,
)
from .shutdown_coordinator import (
    ShutdownCoordinator,
    ShutdownPhase,
    ShutdownState,
    get_shutdown_coordinator,
    reset_shutdown_coordinator,
)
from .synchronization import (
    CrossProcessEvent,
    CrossProcessLock,
    DeadlockDetector,
    EventInfo,
    EventType,
    LockInfo,
    LockState,
    SharedStateManager,
    get_deadlock_detector,
    reset_deadlock_detector,
)

__all__ = [
    # Resource Management
    "ResourceManager",
    "ResourceHandle",
    "ResourceState",
    "ResourceType",
    "CleanupResult",
    "get_resource_manager",
    "reset_resource_manager",
    # Shutdown Coordination
    "ShutdownCoordinator",
    "ShutdownPhase",
    "ShutdownState",
    "get_shutdown_coordinator",
    "reset_shutdown_coordinator",
    # Synchronization
    "CrossProcessLock",
    "CrossProcessEvent",
    "SharedStateManager",
    "DeadlockDetector",
    "LockState",
    "EventType",
    "LockInfo",
    "EventInfo",
    "get_deadlock_detector",
    "reset_deadlock_detector",
]
