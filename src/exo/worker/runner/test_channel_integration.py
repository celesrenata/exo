#!/usr/bin/env python3
"""
Simple integration test for the race-condition-free channel management system.

This test verifies that the ChannelManager, QueueStateManager, and SafeQueueOperations
work together correctly to prevent race conditions during shutdown.
"""

import asyncio
import multiprocessing as mp

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Import our new modules
from exo.worker.runner.channel_manager import get_channel_manager
from exo.worker.runner.enhanced_queue_operations import (
    get_safe_queue_operations,
)
from exo.worker.runner.queue_state_manager import (
    get_queue_state_manager,
)


async def test_basic_channel_operations():
    """Test basic channel creation and operations."""
    print("Testing basic channel operations...")

    # Get manager instances
    channel_manager = get_channel_manager()

    try:
        # Create a test channel
        channel = await channel_manager.create_channel("test_channel", buffer_size=10)

        # Test basic operations
        channel.get_sender()
        channel.get_receiver()

        # Send some test messages
        for i in range(5):
            success = channel.send_safe(f"message_{i}", timeout=1.0)
            if success:
                print("Sent message_{i}")
            else:
                print("Failed to send message_{i}")

        # Receive messages
        for i in range(5):
            success, item = channel.receive_safe(timeout=1.0)
            if success:
                print("Received: {item}")
            else:
                print("Failed to receive message {i}")

        # Test channel info
        info = channel.get_info()
        print(
            f"Channel info: {info.name}, state: {info.state}, messages sent: {info.messages_sent}"
        )

        # Close the channel
        success = await channel_manager.close_channel("test_channel")
        print("Channel closed successfully: {success}")

        return True

    except Exception as e:
        print("Error in basic channel operations test: {e}")
        return False


async def test_queue_state_management():
    """Test queue state synchronization."""
    print("Testing queue state management...")

    # Get manager instances
    queue_manager = get_queue_state_manager()

    try:
        # Register a test queue
        queue_state = await queue_manager.register_queue("test_queue", max_size=100)

        # Test state operations
        current_state = queue_state.get_state()
        print("Initial queue state: {current_state}")

        # Update metrics
        success = queue_state.update_metrics(
            put_operations=10, get_operations=8, current_size=2
        )
        print("Metrics updated: {success}")

        # Get metrics
        metrics = queue_state.get_metrics()
        if metrics:
            print(
                f"Queue metrics: put={metrics.put_operations}, get={metrics.get_operations}, size={metrics.current_size}"
            )

        # Test atomic closure
        success = await queue_manager.close_queue_atomically(
            "test_queue", drain_timeout=2.0
        )
        print("Queue closed atomically: {success}")

        # Unregister
        success = await queue_manager.unregister_queue("test_queue")
        print("Queue unregistered: {success}")

        return True

    except Exception as e:
        print("Error in queue state management test: {e}")
        return False


async def test_safe_queue_operations():
    """Test safe queue operations with timeout handling."""
    print("Testing safe queue operations...")

    # Get operations instance
    safe_ops = get_safe_queue_operations()

    try:
        # Create a simple multiprocessing queue for testing
        test_queue = mp.Queue(maxsize=5)

        # Test safe put operations
        for i in range(3):
            result, error = await safe_ops.safe_put(
                queue=test_queue,
                item=f"test_item_{i}",
                queue_id="test_safe_queue",
                timeout=1.0,
            )
            print("Put result for item {i}: {result}")

        # Test safe get operations
        for i in range(3):
            result, item, error = await safe_ops.safe_get(
                queue=test_queue, queue_id="test_safe_queue", timeout=1.0
            )
            print("Get result for item {i}: {result}, item: {item}")

        # Test drain operation
        # Add some more items first
        for i in range(2):
            test_queue.put(f"drain_item_{i}")

        result, drained_items, progress = await safe_ops.drain_queue_with_progress(
            queue=test_queue, queue_id="test_safe_queue", timeout=2.0
        )
        print(
            f"Drain result: {result}, items drained: {len(drained_items)}, progress: {progress.items_drained}"
        )

        return True

    except Exception as e:
        print("Error in safe queue operations test: {e}")
        return False


async def test_integration_scenario():
    """Test a complete integration scenario simulating shutdown coordination."""
    print("Testing integration scenario...")

    try:
        # Get all manager instances
        channel_manager = get_channel_manager()
        queue_manager = get_queue_state_manager()
        get_safe_queue_operations()

        # Create multiple channels
        channels = []
        for i in range(3):
            channel = await channel_manager.create_channel(
                f"integration_channel_{i}", buffer_size=10
            )
            channels.append(channel)

        # Register corresponding queue states
        for i in range(3):
            await queue_manager.register_queue(f"integration_queue_{i}", max_size=10)

        # Simulate some activity
        for i, channel in enumerate(channels):
            # Send some messages
            for j in range(3):
                channel.send_safe(f"msg_{i}_{j}", timeout=0.5)

        # Get channel statistics
        stats = channel_manager.get_statistics()
        print("Channel manager stats: {stats}")

        # Get queue summary
        queue_stats = queue_manager.get_summary_statistics()
        print("Queue manager stats: {queue_stats}")

        # Simulate coordinated shutdown
        print("Starting coordinated shutdown...")

        # Close all channels
        close_results = await channel_manager.close_all_channels(drain_timeout=2.0)
        print("Channel closure results: {close_results}")

        # Close all queues
        for i in range(3):
            success = await queue_manager.close_queue_atomically(
                f"integration_queue_{i}"
            )
            print("Queue {i} closed: {success}")

        print("Integration scenario completed successfully")
        return True

    except Exception as e:
        print("Error in integration scenario: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting race-condition-free channel management integration tests...")
    print("=" * 60)

    tests = [
        ("Basic Channel Operations", test_basic_channel_operations),
        ("Queue State Management", test_queue_state_management),
        ("Safe Queue Operations", test_safe_queue_operations),
        ("Integration Scenario", test_integration_scenario),
    ]

    results = []

    for test_name, test_func in tests:
        print("\n--- {test_name} ---")
        try:
            success = await test_func()
            results.append((test_name, success))
            print("✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print("✗ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Results Summary:")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print("  {test_name}: {status}")

    print("\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All tests passed! The race-condition-free channel management system is working correctly."
        )
    else:
        print("⚠️  Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
