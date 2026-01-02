"""
Unit tests for ErrorHandler.

Tests error handling and recovery mechanisms, retry logic with backoff,
and specific handlers for different error types.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from exo.worker.runner.error_handler import (
    ErrorContext,
    ErrorHandler,
    ErrorRecord,
    ErrorSeverity,
    RecoveryAction,
    RetryConfig,
    get_error_handler,
    reset_error_handler,
)


class CustomTestError(Exception):
    """Custom exception for testing."""



class TestErrorHandler:
    """Test cases for ErrorHandler."""

    @pytest.fixture
    def handler(self):
        """Create an ErrorHandler instance for testing."""
        return ErrorHandler()

    @pytest.fixture(autouse=True)
    def reset_global_handler(self):
        """Reset global handler after each test."""
        yield
        reset_error_handler()

    def test_initialization(self, handler):
        """Test ErrorHandler initialization."""
        assert len(handler._error_handlers) > 0  # Should have default handlers
        assert len(handler._error_records) == 0
        assert len(handler._retry_configs) == 0
        assert handler._total_errors == 0
        assert handler._resolved_errors == 0
        assert handler._failed_recoveries == 0

    def test_register_handler(self, handler):
        """Test registering custom error handlers."""

        def custom_handler(context):
            return RecoveryAction.RETRY

        handler.register_handler(CustomTestError, custom_handler)

        assert CustomTestError in handler._error_handlers
        assert handler._error_handlers[CustomTestError] == custom_handler

    def test_register_retry_config(self, handler):
        """Test registering retry configurations."""
        config = RetryConfig(max_attempts=5, base_delay=2.0)

        handler.register_retry_config("test_operation", config)

        assert "test_operation" in handler._retry_configs
        assert handler._retry_configs["test_operation"] == config

    @pytest.mark.asyncio
    async def test_handle_error_basic(self, handler):
        """Test basic error handling."""
        error = ValueError("Test error")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        assert isinstance(recovery_action, RecoveryAction)
        assert handler._total_errors == 1
        assert len(handler._error_records) == 1

    @pytest.mark.asyncio
    async def test_handle_error_with_context(self, handler):
        """Test error handling with additional context."""
        error = ValueError("Test error")
        additional_info = {"retry_count": 2, "custom_data": "test"}

        recovery_action = await handler.handle_error(
            error=error,
            component="test_component",
            operation="test_operation",
            runner_id="test_runner",
            additional_info=additional_info,
        )

        assert isinstance(recovery_action, RecoveryAction)

        # Check that error record contains context
        error_record = list(handler._error_records.values())[0]
        assert error_record.context.runner_id == "test_runner"
        assert error_record.context.additional_info == additional_info

    @pytest.mark.asyncio
    async def test_handle_queue_closed_error(self, handler):
        """Test handling of queue closed errors."""
        error = ValueError("Queue is closed")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="normal_operation"
        )

        # Should retry for normal operations
        assert recovery_action == RecoveryAction.RETRY

    @pytest.mark.asyncio
    async def test_handle_queue_closed_error_during_shutdown(self, handler):
        """Test handling of queue closed errors during shutdown."""
        error = ValueError("Queue is closed")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="shutdown_operation"
        )

        # Should skip during shutdown (expected behavior)
        assert recovery_action == RecoveryAction.SKIP

    @pytest.mark.asyncio
    async def test_handle_queue_closed_error_max_retries(self, handler):
        """Test handling of queue closed errors after max retries."""
        error = ValueError("Queue is closed")

        recovery_action = await handler.handle_error(
            error=error,
            component="test_component",
            operation="normal_operation",
            additional_info={"retry_count": 3},
        )

        # Should force cleanup after max retries
        assert recovery_action == RecoveryAction.FORCE_CLEANUP

    @pytest.mark.asyncio
    async def test_handle_closed_resource_error(self, handler):
        """Test handling of closed resource errors."""
        # Mock the ClosedResourceError
        with patch("exo.worker.runner.error_handler.ClosedResourceError", Exception):
            error = Exception("Resource closed")

            recovery_action = await handler.handle_error(
                error=error, component="test_component", operation="normal_operation"
            )

            # Should force cleanup for closed resources during normal operation
            assert recovery_action == RecoveryAction.FORCE_CLEANUP

    @pytest.mark.asyncio
    async def test_handle_closed_resource_error_during_cleanup(self, handler):
        """Test handling of closed resource errors during cleanup."""
        with patch("exo.worker.runner.error_handler.ClosedResourceError", Exception):
            error = Exception("Resource closed")

            recovery_action = await handler.handle_error(
                error=error, component="test_component", operation="cleanup_operation"
            )

            # Should skip during cleanup (expected behavior)
            assert recovery_action == RecoveryAction.SKIP

    @pytest.mark.asyncio
    async def test_handle_timeout_error(self, handler):
        """Test handling of timeout errors."""
        error = asyncio.TimeoutError("Operation timed out")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        # Should retry for single timeout
        assert recovery_action == RecoveryAction.RETRY

    @pytest.mark.asyncio
    async def test_handle_multiple_timeout_errors(self, handler):
        """Test handling of multiple timeout errors."""
        error = asyncio.TimeoutError("Operation timed out")

        # Simulate multiple timeout errors in the same component
        for _i in range(4):
            await handler.handle_error(
                error=error, component="test_component", operation="test_operation"
            )

        # The last one should escalate due to multiple timeouts
        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        assert recovery_action == RecoveryAction.ESCALATE

    @pytest.mark.asyncio
    async def test_handle_connection_error(self, handler):
        """Test handling of connection errors."""
        error = ConnectionError("Connection failed")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        # Should retry for connection errors (usually transient)
        assert recovery_action == RecoveryAction.RETRY

    @pytest.mark.asyncio
    async def test_handle_process_error(self, handler):
        """Test handling of process errors."""
        error = ProcessLookupError("Process not found")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        # Should restart component for process errors
        assert recovery_action == RecoveryAction.RESTART_COMPONENT

    @pytest.mark.asyncio
    async def test_handle_generic_error(self, handler):
        """Test handling of generic/unknown errors."""
        error = CustomTestError("Unknown error")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        # Should escalate for unknown errors
        assert recovery_action == RecoveryAction.ESCALATE

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, handler):
        """Test successful retry with backoff."""
        call_count = 0

        async def operation_that_succeeds_on_second_try():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First attempt fails")
            return "success"

        result = await handler.retry_with_backoff(
            operation_that_succeeds_on_second_try, "test_operation"
        )

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff_all_attempts_fail(self, handler):
        """Test retry with backoff when all attempts fail."""
        call_count = 0

        async def operation_that_always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Attempt {call_count} failed")

        with pytest.raises(RuntimeError, match="Attempt 3 failed"):
            await handler.retry_with_backoff(
                operation_that_always_fails, "test_operation"
            )

        assert call_count == 3  # Default max_attempts
        assert handler._failed_recoveries == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_custom_config(self, handler):
        """Test retry with custom retry configuration."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler.register_retry_config("test_operation", config)

        call_count = 0

        async def operation_that_always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Attempt {call_count} failed")

        with pytest.raises(RuntimeError, match="Attempt 2 failed"):
            await handler.retry_with_backoff(
                operation_that_always_fails, "test_operation"
            )

        assert call_count == 2  # Custom max_attempts

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, handler):
        """Test that exponential backoff increases delay."""
        config = RetryConfig(
            max_attempts=3, base_delay=0.01, exponential_backoff=True, jitter=False
        )
        handler.register_retry_config("test_operation", config)

        delays = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            delays.append(delay)
            await original_sleep(0.001)  # Very short actual delay for testing

        with patch("asyncio.sleep", mock_sleep):

            async def operation_that_always_fails():
                raise RuntimeError("Always fails")

            with pytest.raises(RuntimeError):
                await handler.retry_with_backoff(
                    operation_that_always_fails, "test_operation"
                )

        # Should have exponentially increasing delays
        assert len(delays) == 2  # 3 attempts = 2 delays
        assert delays[1] > delays[0]  # Second delay should be larger

    def test_find_handler_exact_match(self, handler):
        """Test finding handler with exact type match."""

        def custom_handler(context):
            return RecoveryAction.RETRY

        handler.register_handler(CustomTestError, custom_handler)

        found_handler = handler._find_handler(CustomTestError)
        assert found_handler == custom_handler

    def test_find_handler_parent_class_match(self, handler):
        """Test finding handler with parent class match."""
        # ValueError should match Exception handler
        found_handler = handler._find_handler(ValueError)

        # Should find the ValueError-specific handler or Exception handler
        assert found_handler is not None

    def test_find_handler_fallback(self, handler):
        """Test fallback to generic handler."""
        found_handler = handler._find_handler(CustomTestError)

        # Should fallback to generic Exception handler
        assert found_handler is not None

    def test_determine_severity(self, handler):
        """Test severity determination based on recovery action."""
        assert (
            handler._determine_severity(Exception(), RecoveryAction.SHUTDOWN)
            == ErrorSeverity.CRITICAL
        )
        assert (
            handler._determine_severity(Exception(), RecoveryAction.ESCALATE)
            == ErrorSeverity.HIGH
        )
        assert (
            handler._determine_severity(Exception(), RecoveryAction.RESTART_COMPONENT)
            == ErrorSeverity.HIGH
        )
        assert (
            handler._determine_severity(Exception(), RecoveryAction.FORCE_CLEANUP)
            == ErrorSeverity.MEDIUM
        )
        assert (
            handler._determine_severity(Exception(), RecoveryAction.RETRY)
            == ErrorSeverity.LOW
        )
        assert (
            handler._determine_severity(Exception(), RecoveryAction.SKIP)
            == ErrorSeverity.LOW
        )

    @pytest.mark.asyncio
    async def test_error_record_creation(self, handler):
        """Test that error records are created properly."""
        error = ValueError("Test error")

        await handler.handle_error(
            error=error,
            component="test_component",
            operation="test_operation",
            runner_id="test_runner",
        )

        assert len(handler._error_records) == 1

        error_record = list(handler._error_records.values())[0]
        assert isinstance(error_record, ErrorRecord)
        assert error_record.context.error == error
        assert error_record.context.component == "test_component"
        assert error_record.context.operation == "test_operation"
        assert error_record.context.runner_id == "test_runner"
        assert error_record.retry_count == 0
        assert error_record.resolved is False

    def test_get_error_statistics(self, handler):
        """Test getting error statistics."""
        # Initially empty
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 0
        assert stats["resolved_errors"] == 0
        assert stats["failed_recoveries"] == 0
        assert stats["active_errors"] == 0
        assert stats["success_rate"] == 0.0

        # Add some errors
        handler._total_errors = 10
        handler._resolved_errors = 7
        handler._failed_recoveries = 2

        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 10
        assert stats["resolved_errors"] == 7
        assert stats["failed_recoveries"] == 2
        assert stats["success_rate"] == 0.7

    def test_get_recent_errors(self, handler):
        """Test getting recent errors."""
        # Create some error records with different timestamps
        old_error = ErrorRecord(
            error_id="old_error",
            context=ErrorContext(
                error=ValueError("Old error"),
                component="test",
                operation="test",
                timestamp=datetime.now() - timedelta(hours=2),
            ),
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
        )

        recent_error = ErrorRecord(
            error_id="recent_error",
            context=ErrorContext(
                error=ValueError("Recent error"),
                component="test",
                operation="test",
                timestamp=datetime.now() - timedelta(minutes=30),
            ),
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
        )

        handler._error_records["old_error"] = old_error
        handler._error_records["recent_error"] = recent_error

        recent_errors = handler.get_recent_errors(hours=1)

        assert len(recent_errors) == 1
        assert recent_errors[0].error_id == "recent_error"

    def test_mark_error_resolved(self, handler):
        """Test marking errors as resolved."""
        error_record = ErrorRecord(
            error_id="test_error",
            context=ErrorContext(
                error=ValueError("Test error"), component="test", operation="test"
            ),
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
        )

        handler._error_records["test_error"] = error_record

        handler.mark_error_resolved("test_error")

        assert error_record.resolved is True
        assert error_record.resolution_time is not None
        assert handler._resolved_errors == 1

    def test_mark_error_resolved_nonexistent(self, handler):
        """Test marking non-existent error as resolved."""
        # Should not raise exception
        handler.mark_error_resolved("nonexistent_error")
        assert handler._resolved_errors == 0

    def test_clear_old_errors(self, handler):
        """Test clearing old error records."""
        # Create old and recent error records
        old_error = ErrorRecord(
            error_id="old_error",
            context=ErrorContext(
                error=ValueError("Old error"),
                component="test",
                operation="test",
                timestamp=datetime.now() - timedelta(hours=25),
            ),
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
        )

        recent_error = ErrorRecord(
            error_id="recent_error",
            context=ErrorContext(
                error=ValueError("Recent error"),
                component="test",
                operation="test",
                timestamp=datetime.now() - timedelta(hours=1),
            ),
            severity=ErrorSeverity.LOW,
            recovery_action=RecoveryAction.RETRY,
        )

        handler._error_records["old_error"] = old_error
        handler._error_records["recent_error"] = recent_error

        handler.clear_old_errors(hours=24)

        assert "old_error" not in handler._error_records
        assert "recent_error" in handler._error_records

    def test_global_handler(self):
        """Test global handler instance management."""
        # Get global instance
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        # Should be the same instance
        assert handler1 is handler2

        # Reset and get new instance
        reset_error_handler()
        handler3 = get_error_handler()

        # Should be different instance
        assert handler3 is not handler1

    @pytest.mark.asyncio
    async def test_handler_exception_handling(self, handler):
        """Test that exceptions in handlers are handled gracefully."""

        def failing_handler(context):
            raise RuntimeError("Handler failed")

        handler.register_handler(CustomTestError, failing_handler)

        error = CustomTestError("Test error")

        recovery_action = await handler.handle_error(
            error=error, component="test_component", operation="test_operation"
        )

        # Should escalate when handler fails
        assert recovery_action == RecoveryAction.ESCALATE
