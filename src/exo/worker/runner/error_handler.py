"""
Enhanced error handling for runner supervisor with specific handlers for queue closure errors,
retry mechanisms, and graceful degradation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from loguru import logger


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Actions to take for error recovery."""

    RETRY = "retry"
    SKIP = "skip"
    FORCE_CLEANUP = "force_cleanup"
    ESCALATE = "escalate"
    RESTART_COMPONENT = "restart_component"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorContext:
    """Context information for an error."""

    error: Exception
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    runner_id: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    context: ErrorContext
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ErrorHandler:
    """
    Enhanced error handler with specific handlers for different error types,
    retry mechanisms, and graceful degradation strategies.
    """

    def __init__(self):
        """Initialize the error handler."""
        self._error_handlers: Dict[Type[Exception], Callable] = {}
        self._error_records: Dict[str, ErrorRecord] = {}
        self._retry_configs: Dict[str, RetryConfig] = {}
        self._default_retry_config = RetryConfig()

        # Register default error handlers
        self._register_default_handlers()

        # Statistics
        self._total_errors = 0
        self._resolved_errors = 0
        self._failed_recoveries = 0

        logger.debug("Initialized ErrorHandler")

    def _register_default_handlers(self):
        """Register default error handlers for common error types."""

        # Queue closure errors
        self.register_handler(ValueError, self._handle_queue_closed_error)

        # Resource errors
        from anyio import BrokenResourceError, ClosedResourceError

        self.register_handler(ClosedResourceError, self._handle_closed_resource_error)
        self.register_handler(BrokenResourceError, self._handle_broken_resource_error)

        # Timeout errors
        self.register_handler(asyncio.TimeoutError, self._handle_timeout_error)
        self.register_handler(TimeoutError, self._handle_timeout_error)

        # Connection errors
        self.register_handler(ConnectionError, self._handle_connection_error)

        # Process errors
        self.register_handler(ProcessLookupError, self._handle_process_error)

        # Generic exception handler (fallback)
        self.register_handler(Exception, self._handle_generic_error)

    def register_handler(self, error_type: Type[Exception], handler: Callable):
        """
        Register a handler for a specific error type.

        Args:
            error_type: Type of exception to handle
            handler: Handler function that takes (ErrorContext) and returns RecoveryAction
        """
        self._error_handlers[error_type] = handler
        logger.debug(f"Registered error handler for {error_type.__name__}")

    def register_retry_config(self, operation: str, config: RetryConfig):
        """
        Register retry configuration for a specific operation.

        Args:
            operation: Name of the operation
            config: Retry configuration
        """
        self._retry_configs[operation] = config
        logger.debug(f"Registered retry config for operation: {operation}")

    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        runner_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """
        Handle an error with appropriate recovery action.

        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation that failed
            runner_id: Optional runner ID for context
            additional_info: Additional context information

        Returns:
            RecoveryAction to take
        """
        self._total_errors += 1

        # Create error context
        context = ErrorContext(
            error=error,
            component=component,
            operation=operation,
            runner_id=runner_id,
            additional_info=additional_info or {},
        )

        # Find appropriate handler
        handler = self._find_handler(type(error))

        try:
            # Execute handler
            recovery_action = await handler(context)

            # Record the error
            error_record = ErrorRecord(
                error_id=uuid4().hex[:8],
                context=context,
                severity=self._determine_severity(error, recovery_action),
                recovery_action=recovery_action,
            )

            self._error_records[error_record.error_id] = error_record

            # Log the error and recovery action
            self._log_error(error_record)

            return recovery_action

        except Exception as handler_error:
            logger.error(f"Error in error handler: {handler_error}")
            return RecoveryAction.ESCALATE

    async def retry_with_backoff(
        self, operation: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: Async function to retry
            operation_name: Name of the operation for logging
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        retry_config = self._retry_configs.get(
            operation_name, self._default_retry_config
        )

        last_error = None

        for attempt in range(retry_config.max_attempts):
            try:
                logger.debug(
                    f"Attempting {operation_name} (attempt {attempt + 1}/{retry_config.max_attempts})"
                )
                result = await operation(*args, **kwargs)

                if attempt > 0:
                    logger.info(
                        f"Operation {operation_name} succeeded after {attempt + 1} attempts"
                    )

                return result

            except Exception as e:
                last_error = e

                if attempt < retry_config.max_attempts - 1:
                    # Calculate delay
                    delay = retry_config.base_delay

                    if retry_config.exponential_backoff:
                        delay *= 2**attempt

                    delay = min(delay, retry_config.max_delay)

                    if retry_config.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

                    logger.warning(
                        f"Operation {operation_name} failed (attempt {attempt + 1}): {e}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation {operation_name} failed after {retry_config.max_attempts} attempts: {e}"
                    )

        # All attempts failed
        self._failed_recoveries += 1
        raise last_error

    def _find_handler(self, error_type: Type[Exception]) -> Callable:
        """Find the most specific handler for an error type."""

        # Look for exact match first
        if error_type in self._error_handlers:
            return self._error_handlers[error_type]

        # Look for parent class matches
        for registered_type, handler in self._error_handlers.items():
            if issubclass(error_type, registered_type):
                return handler

        # Fallback to generic handler
        return self._error_handlers.get(Exception, self._handle_generic_error)

    def _determine_severity(
        self, error: Exception, recovery_action: RecoveryAction
    ) -> ErrorSeverity:
        """Determine the severity of an error based on type and recovery action."""

        if recovery_action == RecoveryAction.SHUTDOWN:
            return ErrorSeverity.CRITICAL
        elif recovery_action in (
            RecoveryAction.ESCALATE,
            RecoveryAction.RESTART_COMPONENT,
        ):
            return ErrorSeverity.HIGH
        elif recovery_action == RecoveryAction.FORCE_CLEANUP:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _log_error(self, error_record: ErrorRecord):
        """Log an error record with appropriate level."""

        context = error_record.context
        log_msg = (
            f"Error in {context.component}.{context.operation}: {context.error}. "
            f"Recovery action: {error_record.recovery_action.value}"
        )

        if context.runner_id:
            log_msg = f"[Runner {context.runner_id}] {log_msg}"

        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    # Specific error handlers

    async def _handle_queue_closed_error(self, context: ErrorContext) -> RecoveryAction:
        """Handle queue closure errors specifically."""

        error_msg = str(context.error).lower()

        if "queue is closed" in error_msg:
            logger.debug(f"Detected queue closure error in {context.component}")

            # If this is during shutdown, it's expected
            if "shutdown" in context.operation.lower():
                return RecoveryAction.SKIP

            # If this is during normal operation, try to recover
            if context.additional_info.get("retry_count", 0) < 2:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.FORCE_CLEANUP

        # Other ValueError types
        return RecoveryAction.RETRY

    async def _handle_closed_resource_error(
        self, context: ErrorContext
    ) -> RecoveryAction:
        """Handle closed resource errors."""

        logger.debug(f"Detected closed resource error in {context.component}")

        # During shutdown, this is expected
        if (
            "shutdown" in context.operation.lower()
            or "cleanup" in context.operation.lower()
        ):
            return RecoveryAction.SKIP

        # During normal operation, this indicates a problem
        return RecoveryAction.FORCE_CLEANUP

    async def _handle_broken_resource_error(
        self, context: ErrorContext
    ) -> RecoveryAction:
        """Handle broken resource errors."""

        logger.debug(f"Detected broken resource error in {context.component}")

        # Broken resources usually need component restart
        return RecoveryAction.RESTART_COMPONENT

    async def _handle_timeout_error(self, context: ErrorContext) -> RecoveryAction:
        """Handle timeout errors."""

        logger.debug(f"Detected timeout error in {context.component}")

        # Check if this is a recurring timeout
        recent_timeouts = sum(
            1
            for record in self._error_records.values()
            if (
                isinstance(record.context.error, (asyncio.TimeoutError, TimeoutError))
                and record.context.component == context.component
                and (datetime.now() - record.context.timestamp).total_seconds()
                < 300  # 5 minutes
            )
        )

        if recent_timeouts > 3:
            logger.warning(f"Multiple timeouts in {context.component}, escalating")
            return RecoveryAction.ESCALATE

        return RecoveryAction.RETRY

    async def _handle_connection_error(self, context: ErrorContext) -> RecoveryAction:
        """Handle connection errors."""

        logger.debug(f"Detected connection error in {context.component}")

        # Connection errors are usually transient
        return RecoveryAction.RETRY

    async def _handle_process_error(self, context: ErrorContext) -> RecoveryAction:
        """Handle process-related errors."""

        logger.debug(f"Detected process error in {context.component}")

        # Process errors usually indicate the process is gone
        return RecoveryAction.RESTART_COMPONENT

    async def _handle_generic_error(self, context: ErrorContext) -> RecoveryAction:
        """Handle generic errors (fallback handler)."""

        logger.debug(
            f"Handling generic error in {context.component}: {type(context.error).__name__}"
        )

        # For unknown errors, be conservative
        return RecoveryAction.ESCALATE

    # Utility methods

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""

        active_errors = len([r for r in self._error_records.values() if not r.resolved])

        return {
            "total_errors": self._total_errors,
            "resolved_errors": self._resolved_errors,
            "failed_recoveries": self._failed_recoveries,
            "active_errors": active_errors,
            "success_rate": self._resolved_errors / max(self._total_errors, 1),
        }

    def get_recent_errors(self, hours: int = 1) -> List[ErrorRecord]:
        """Get errors from the last N hours."""

        cutoff = datetime.now() - timedelta(hours=hours)

        return [
            record
            for record in self._error_records.values()
            if record.context.timestamp > cutoff
        ]

    def mark_error_resolved(self, error_id: str):
        """Mark an error as resolved."""

        if error_id in self._error_records:
            record = self._error_records[error_id]
            record.resolved = True
            record.resolution_time = datetime.now()
            self._resolved_errors += 1

            logger.debug(f"Marked error {error_id} as resolved")

    def clear_old_errors(self, hours: int = 24):
        """Clear error records older than specified hours."""

        cutoff = datetime.now() - timedelta(hours=hours)

        old_errors = [
            error_id
            for error_id, record in self._error_records.items()
            if record.context.timestamp < cutoff
        ]

        for error_id in old_errors:
            del self._error_records[error_id]

        logger.debug(f"Cleared {len(old_errors)} old error records")


# Global instance for cross-module usage
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def reset_error_handler() -> None:
    """Reset the global error handler (mainly for testing)."""
    global _global_error_handler
    _global_error_handler = None
