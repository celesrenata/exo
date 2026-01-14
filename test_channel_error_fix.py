#!/usr/bin/env python3
"""
Test script to verify the ClosedResourceError fix in runner supervisor.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from anyio import ClosedResourceError
from unittest.mock import Mock, AsyncMock
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_closed_resource_error_handling():
    """Test that ClosedResourceError is handled gracefully without crashing."""

    print("Testing ClosedResourceError handling in runner supervisor...")

    try:
        # Import the runner supervisor
        from exo.worker.runner.runner_supervisor import RunnerSupervisor

        # Create mock objects
        mock_bound_instance = Mock()
        mock_bound_instance.bound_runner_id = "test_runner_123"

        mock_event_sender = Mock()
        mock_task_receiver = Mock()
        mock_logger = Mock()

        # Create a supervisor instance
        supervisor = RunnerSupervisor(
            bound_instance=mock_bound_instance,
            event_sender=mock_event_sender,
            task_receiver=mock_task_receiver,
            _logger=mock_logger,
        )

        # Mock the event receiver to raise ClosedResourceError
        mock_ev_recv = Mock()
        mock_ev_recv.__enter__ = Mock(return_value=mock_ev_recv)
        mock_ev_recv.__exit__ = Mock(return_value=None)
        mock_ev_recv.receive_async = AsyncMock(side_effect=ClosedResourceError())

        supervisor._ev_recv = mock_ev_recv
        supervisor._shutdown_in_progress = False
        supervisor.pending = {}

        # Mock error handler
        supervisor._error_handler = Mock()
        supervisor._error_handler.handle_error = AsyncMock()

        # Test that _forward_events handles ClosedResourceError gracefully
        try:
            await supervisor._forward_events()
            print("✅ ClosedResourceError handled gracefully - no crash!")
            return True
        except ClosedResourceError:
            print("❌ ClosedResourceError was not caught properly")
            return False
        except Exception as e:
            print("❌ Unexpected error: {e}")
            return False

    except ImportError as e:
        print("❌ Import error: {e}")
        return False
    except Exception as e:
        print("❌ Test setup error: {e}")
        return False


async def main():
    """Run the test."""
    print("Starting ClosedResourceError fix validation...")

    success = await test_closed_resource_error_handling()

    if success:
        print("\n🎉 Test PASSED: ClosedResourceError fix is working correctly!")
        return 0
    else:
        print("\n💥 Test FAILED: ClosedResourceError fix needs more work")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
