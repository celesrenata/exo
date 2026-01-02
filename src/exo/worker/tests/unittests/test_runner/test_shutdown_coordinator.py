"""
Unit tests for ShutdownCoordinator.

Tests the three-phase shutdown protocol, timeout handling, cross-process coordination,
and error recovery mechanisms.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.runner.shutdown_coordinator import (
    ShutdownCoordinator,
    ShutdownPhase,
    ShutdownState,
    get_shutdown_coordinator,
    reset_shutdown_coordinator,
)


class TestShutdownCoordinator:
    """Test cases for ShutdownCoordinator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def coordinator(self, temp_dir):
        """Create a ShutdownCoordinator instance for testing."""
        return ShutdownCoordinator(temp_dir=temp_dir)
    
    @pytest.fixture(autouse=True)
    def reset_global_coordinator(self):
        """Reset global coordinator after each test."""
        yield
        reset_shutdown_coordinator()
    
    def test_initialization(self, temp_dir):
        """Test ShutdownCoordinator initialization."""
        coordinator = ShutdownCoordinator(temp_dir=temp_dir)
        
        assert coordinator._temp_dir == temp_dir
        assert coordinator._coordination_dir.exists()
        assert len(coordinator._shutdown_states) == 0
        assert len(coordinator._shutdown_handlers) == 0
    
    def test_initialization_default_temp_dir(self):
        """Test initialization with default temp directory."""
        coordinator = ShutdownCoordinator()
        
        assert coordinator._temp_dir is not None
        assert coordinator._coordination_dir.exists()
    
    @pytest.mark.asyncio
    async def test_successful_shutdown(self, coordinator):
        """Test successful three-phase shutdown."""
        runner_id = "test_runner_1"
        
        # Mock the phase methods to succeed
        coordinator._phase_signaling = AsyncMock(return_value=True)
        coordinator._phase_draining = AsyncMock(return_value=True)
        coordinator._phase_closing = AsyncMock(return_value=True)
        
        result = await coordinator.initiate_shutdown(runner_id, timeout=10.0)
        
        assert result is True
        
        # Verify phases were called
        coordinator._phase_signaling.assert_called_once_with(runner_id)
        coordinator._phase_draining.assert_called_once_with(runner_id)
        coordinator._phase_closing.assert_called_once_with(runner_id)
        
        # Verify final state
        state = coordinator.get_shutdown_status(runner_id)
        assert state is None  # Should be cleaned up after completion
    
    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, coordinator):
        """Test shutdown with timeout."""
        runner_id = "test_runner_timeout"
        
        # Mock phase to take too long
        async def slow_phase(runner_id):
            await asyncio.sleep(2.0)
            return True
        
        coordinator._phase_signaling = AsyncMock(return_value=True)
        coordinator._phase_draining = slow_phase
        coordinator._phase_closing = AsyncMock(return_value=True)
        
        result = await coordinator.initiate_shutdown(runner_id, timeout=0.5)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_shutdown_phase_failure(self, coordinator):
        """Test shutdown when a phase fails."""
        runner_id = "test_runner_fail"
        
        # Mock signaling phase to fail
        coordinator._phase_signaling = AsyncMock(return_value=False)
        coordinator._phase_draining = AsyncMock(return_value=True)
        coordinator._phase_closing = AsyncMock(return_value=True)
        
        result = await coordinator.initiate_shutdown(runner_id, timeout=10.0)
        
        assert result is False
        
        # Only signaling should have been called
        coordinator._phase_signaling.assert_called_once_with(runner_id)
        coordinator._phase_draining.assert_not_called()
        coordinator._phase_closing.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_shutdown_same_runner(self, coordinator):
        """Test concurrent shutdown requests for the same runner."""
        runner_id = "test_runner_concurrent"
        
        # Mock phases to take some time
        async def slow_phase(runner_id):
            await asyncio.sleep(0.1)
            return True
        
        coordinator._phase_signaling = slow_phase
        coordinator._phase_draining = slow_phase
        coordinator._phase_closing = slow_phase
        
        # Start two concurrent shutdowns
        task1 = asyncio.create_task(coordinator.initiate_shutdown(runner_id, timeout=5.0))
        task2 = asyncio.create_task(coordinator.initiate_shutdown(runner_id, timeout=5.0))
        
        result1, result2 = await asyncio.gather(task1, task2)
        
        # Both should succeed (second one waits for first)
        assert result1 is True
        assert result2 is True
    
    @pytest.mark.asyncio
    async def test_phase_signaling(self, coordinator):
        """Test the signaling phase."""
        runner_id = "test_runner_signaling"
        
        # Create shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.SIGNALING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        coordinator._signal_files[runner_id] = coordinator._coordination_dir / f"test_signal_{runner_id}.signal"
        
        # Add a mock handler
        handler_mock = MagicMock()
        coordinator.register_shutdown_handler(handler_mock)
        
        result = await coordinator._phase_signaling(runner_id)
        
        assert result is True
        assert state.phase == ShutdownPhase.SIGNALING
        
        # Verify signal file was created
        signal_file = coordinator._signal_files[runner_id]
        assert signal_file.exists()
        content = signal_file.read_text()
        assert content.startswith("SHUTDOWN_SIGNALING:")
        
        # Verify handler was called
        handler_mock.assert_called_once_with(runner_id)
    
    @pytest.mark.asyncio
    async def test_phase_draining(self, coordinator):
        """Test the draining phase."""
        runner_id = "test_runner_draining"
        
        # Create shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.SIGNALING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        coordinator._signal_files[runner_id] = coordinator._coordination_dir / f"test_signal_{runner_id}.signal"
        
        result = await coordinator._phase_draining(runner_id)
        
        assert result is True
        assert state.phase == ShutdownPhase.DRAINING
        
        # Verify signal file was updated
        signal_file = coordinator._signal_files[runner_id]
        content = signal_file.read_text()
        assert content.startswith("SHUTDOWN_DRAINING:")
    
    @pytest.mark.asyncio
    async def test_phase_closing(self, coordinator):
        """Test the closing phase."""
        runner_id = "test_runner_closing"
        
        # Create shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.DRAINING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        coordinator._signal_files[runner_id] = coordinator._coordination_dir / f"test_signal_{runner_id}.signal"
        
        result = await coordinator._phase_closing(runner_id)
        
        assert result is True
        assert state.phase == ShutdownPhase.CLOSING
        
        # Verify signal file was updated
        signal_file = coordinator._signal_files[runner_id]
        content = signal_file.read_text()
        assert content.startswith("SHUTDOWN_CLOSING:")
    
    @pytest.mark.asyncio
    async def test_wait_for_shutdown_complete(self, coordinator):
        """Test waiting for shutdown completion."""
        runner_id = "test_runner_wait"
        
        # Create shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.DRAINING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        # Start waiting task
        wait_task = asyncio.create_task(coordinator.wait_for_shutdown_complete(runner_id))
        
        # Give it a moment to start waiting
        await asyncio.sleep(0.1)
        
        # Mark as complete
        state.phase = ShutdownPhase.COMPLETE
        
        result = await wait_task
        assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_shutdown_timeout(self, coordinator):
        """Test waiting for shutdown with timeout."""
        runner_id = "test_runner_wait_timeout"
        
        # Create shutdown state with short timeout
        state = ShutdownState(
            phase=ShutdownPhase.DRAINING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=0.2),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        result = await coordinator.wait_for_shutdown_complete(runner_id)
        
        assert result is False
        assert state.phase == ShutdownPhase.FAILED
    
    def test_register_shutdown_handler(self, coordinator):
        """Test registering shutdown handlers."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        coordinator.register_shutdown_handler(handler1)
        coordinator.register_shutdown_handler(handler2)
        
        assert len(coordinator._shutdown_handlers) == 2
        assert handler1 in coordinator._shutdown_handlers
        assert handler2 in coordinator._shutdown_handlers
    
    def test_get_shutdown_status(self, coordinator):
        """Test getting shutdown status."""
        runner_id = "test_runner_status"
        
        # No status initially
        status = coordinator.get_shutdown_status(runner_id)
        assert status is None
        
        # Add shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.DRAINING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        status = coordinator.get_shutdown_status(runner_id)
        assert status == state
        assert status.phase == ShutdownPhase.DRAINING
    
    def test_is_shutdown_in_progress(self, coordinator):
        """Test checking if shutdown is in progress."""
        runner_id = "test_runner_progress"
        
        # No shutdown initially
        assert coordinator.is_shutdown_in_progress(runner_id) is False
        
        # Add active shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.DRAINING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        assert coordinator.is_shutdown_in_progress(runner_id) is True
        
        # Complete shutdown
        state.phase = ShutdownPhase.COMPLETE
        assert coordinator.is_shutdown_in_progress(runner_id) is False
        
        # Failed shutdown
        state.phase = ShutdownPhase.FAILED
        assert coordinator.is_shutdown_in_progress(runner_id) is False
    
    def test_check_shutdown_signal(self, coordinator):
        """Test checking for shutdown signals from other processes."""
        runner_id = "test_runner_signal"
        
        # No signal initially
        signal = coordinator.check_shutdown_signal(runner_id)
        assert signal is None
        
        # Create signal file
        signal_file = coordinator._coordination_dir / f"shutdown_{runner_id}_test123.signal"
        signal_file.write_text("SHUTDOWN_DRAINING:1234567890")
        
        signal = coordinator.check_shutdown_signal(runner_id)
        assert signal == ShutdownPhase.DRAINING
        
        # Test invalid signal file
        signal_file.write_text("INVALID_CONTENT")
        signal = coordinator.check_shutdown_signal(runner_id)
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_cleanup_coordination_files(self, coordinator):
        """Test cleanup of coordination files."""
        runner_id = "test_runner_cleanup"
        
        # Create signal file
        signal_file = coordinator._coordination_dir / f"shutdown_{runner_id}_test123.signal"
        signal_file.write_text("SHUTDOWN_SIGNALING:1234567890")
        coordinator._signal_files[runner_id] = signal_file
        
        # Add shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.COMPLETE,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        # Cleanup
        await coordinator._cleanup_coordination_files(runner_id)
        
        # Verify cleanup
        assert not signal_file.exists()
        assert runner_id not in coordinator._signal_files
        assert runner_id not in coordinator._shutdown_states
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self, coordinator):
        """Test cleanup of all coordination state."""
        runner_id1 = "test_runner_1"
        runner_id2 = "test_runner_2"
        
        # Create signal files and states
        for runner_id in [runner_id1, runner_id2]:
            signal_file = coordinator._coordination_dir / f"shutdown_{runner_id}_test.signal"
            signal_file.write_text("SHUTDOWN_SIGNALING:1234567890")
            coordinator._signal_files[runner_id] = signal_file
            
            state = ShutdownState(
                phase=ShutdownPhase.COMPLETE,
                started_at=datetime.now(),
                timeout_at=datetime.now() + timedelta(seconds=30),
                runner_id=runner_id
            )
            coordinator._shutdown_states[runner_id] = state
        
        # Add handlers
        handler = MagicMock()
        coordinator.register_shutdown_handler(handler)
        
        # Cleanup all
        await coordinator.cleanup_all()
        
        # Verify cleanup
        assert len(coordinator._signal_files) == 0
        assert len(coordinator._shutdown_states) == 0
        assert len(coordinator._shutdown_handlers) == 0
    
    def test_global_coordinator(self):
        """Test global coordinator instance management."""
        # Get global instance
        coordinator1 = get_shutdown_coordinator()
        coordinator2 = get_shutdown_coordinator()
        
        # Should be the same instance
        assert coordinator1 is coordinator2
        
        # Reset and get new instance
        reset_shutdown_coordinator()
        coordinator3 = get_shutdown_coordinator()
        
        # Should be different instance
        assert coordinator3 is not coordinator1
    
    @pytest.mark.asyncio
    async def test_error_handling_in_phases(self, coordinator):
        """Test error handling during phase execution."""
        runner_id = "test_runner_error"
        
        # Mock phase to raise exception
        coordinator._phase_signaling = AsyncMock(side_effect=Exception("Test error"))
        
        result = await coordinator.initiate_shutdown(runner_id, timeout=10.0)
        
        assert result is False
        
        # Verify error was recorded
        state = coordinator.get_shutdown_status(runner_id)
        assert state is None  # Should be cleaned up after error
    
    @pytest.mark.asyncio
    async def test_signal_file_creation_error(self, coordinator):
        """Test handling of signal file creation errors."""
        runner_id = "test_runner_signal_error"
        
        # Create shutdown state
        state = ShutdownState(
            phase=ShutdownPhase.SIGNALING,
            started_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30),
            runner_id=runner_id
        )
        coordinator._shutdown_states[runner_id] = state
        
        # Set invalid signal file path
        invalid_path = Path("/invalid/path/signal.file")
        coordinator._signal_files[runner_id] = invalid_path
        
        result = await coordinator._phase_signaling(runner_id)
        
        assert result is False
        assert state.error_count > 0
        assert state.last_error is not None