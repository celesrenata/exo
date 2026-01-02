"""
Example demonstrating the integration of shutdown coordination infrastructure.

This shows how the ShutdownCoordinator, ResourceManager, and synchronization
primitives work together to prevent race conditions during runner shutdown.
"""

import asyncio
import multiprocessing as mp

from .resource_manager import ResourceType, get_resource_manager
from .shutdown_coordinator import get_shutdown_coordinator
from .synchronization import CrossProcessLock, CrossProcessEvent, EventType


async def example_coordinated_shutdown():
    """
    Example of coordinated shutdown using the new infrastructure.
    
    This demonstrates the three-phase shutdown protocol with proper
    resource cleanup ordering.
    """
    # Get global instances
    coordinator = get_shutdown_coordinator()
    resource_manager = get_resource_manager()
    
    # Example runner ID
    runner_id = "example_runner_001"
    
    # Register some example resources (simulating queues, channels, etc.)
    
    # Register a multiprocessing queue (high cleanup order - cleaned up last)
    example_queue = mp.Queue()
    queue_handle = resource_manager.register_resource(
        resource=example_queue,
        resource_type=ResourceType.QUEUE,
        cleanup_order=100,  # Higher numbers cleaned up later
        cleanup_func=lambda: example_queue.close(),
        resource_id="main_queue"
    )
    
    # Register a cross-process lock (medium cleanup order)
    example_lock = CrossProcessLock("example_lock")
    def cleanup_lock():
        if example_lock.is_acquired():
            example_lock.release()
    
    lock_handle = resource_manager.register_resource(
        resource=example_lock,
        resource_type=ResourceType.CUSTOM,
        cleanup_order=50,
        cleanup_func=cleanup_lock,
        resource_id="coordination_lock",
        dependencies={"main_queue"}  # Depends on queue being available
    )
    
    # Register a cross-process event (low cleanup order - cleaned up first)
    example_event = CrossProcessEvent("shutdown_event")
    def cleanup_event():
        example_event.clear_event()
    
    event_handle = resource_manager.register_resource(
        resource=example_event,
        resource_type=ResourceType.CUSTOM,
        cleanup_order=10,
        cleanup_func=cleanup_event,
        resource_id="shutdown_event"
    )
    
    print(f"Registered {len(resource_manager.get_all_resources())} resources")
    
    # Register a shutdown handler
    def shutdown_handler(runner_id: str):
        print(f"Shutdown handler called for runner {runner_id}")
        # Signal shutdown event to other processes
        example_event.signal(EventType.SHUTDOWN_SIGNAL, {"runner_id": runner_id})
    
    coordinator.register_shutdown_handler(shutdown_handler)
    
    # Simulate some work
    print("Simulating runner work...")
    await asyncio.sleep(1.0)
    
    # Initiate coordinated shutdown
    print(f"Initiating coordinated shutdown for {runner_id}")
    success = await coordinator.initiate_shutdown(runner_id, timeout=30.0)
    
    if success:
        print("Shutdown completed successfully")
        
        # Clean up resources using ResourceManager
        print("Cleaning up resources...")
        cleanup_result = await resource_manager.cleanup_resources(timeout=20.0)
        
        if cleanup_result.success:
            print(f"Resource cleanup successful: {len(cleanup_result.cleaned_resources)} resources cleaned")
        else:
            print(f"Resource cleanup had issues: {len(cleanup_result.failed_resources)} failed")
            for error in cleanup_result.errors:
                print(f"  Error: {error}")
    else:
        print("Shutdown failed or timed out")
    
    # Cleanup coordination state
    await coordinator.cleanup_all()


def example_cross_process_coordination():
    """
    Example of cross-process coordination using locks and events.
    
    This demonstrates how processes can coordinate safely during shutdown.
    """
    # Create coordination primitives
    shutdown_lock = CrossProcessLock("shutdown_coordination")
    shutdown_event = CrossProcessEvent("process_shutdown")
    
    print("Attempting to acquire shutdown coordination lock...")
    
    # Use context manager for safe lock handling
    with shutdown_lock.acquire_context(timeout=10.0) as acquired:
        if acquired:
            print("Lock acquired, performing coordinated shutdown...")
            
            # Signal other processes about shutdown
            shutdown_event.signal(
                EventType.SHUTDOWN_SIGNAL,
                {"message": "Coordinated shutdown initiated", "process": mp.current_process().pid}
            )
            
            # Simulate shutdown work
            import time
            time.sleep(2.0)
            
            print("Shutdown work completed, releasing lock")
        else:
            print("Failed to acquire lock, checking for shutdown signals...")
            
            # Check if another process signaled shutdown
            event_info = shutdown_event.check_event(EventType.SHUTDOWN_SIGNAL)
            if event_info:
                print(f"Received shutdown signal from process {event_info.source_process}")
                print(f"Message: {event_info.data.get('message', 'No message')}")
            else:
                print("No shutdown signals detected")


if __name__ == "__main__":
    print("=== Shutdown Coordination Infrastructure Example ===")
    
    print("\n1. Cross-process coordination example:")
    example_cross_process_coordination()
    
    print("\n2. Coordinated shutdown example:")
    asyncio.run(example_coordinated_shutdown())
    
    print("\nExample completed successfully!")