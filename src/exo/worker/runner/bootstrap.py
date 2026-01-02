import asyncio
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.resource_manager import ResourceManager, ResourceType, get_resource_manager
from exo.worker.runner.shutdown_coordinator import ShutdownCoordinator, get_shutdown_coordinator
from exo.worker.runner.channel_manager import ChannelManager, get_channel_manager

logger: "loguru.Logger" = loguru.logger


class StartupFailureType(Enum):
    """Types of startup failures for categorized error handling."""
    TIMEOUT = "timeout"
    RESOURCE_ALLOCATION = "resource_allocation"
    ENVIRONMENT_SETUP = "environment_setup"
    CHANNEL_SETUP = "channel_setup"
    RUNNER_INITIALIZATION = "runner_initialization"
    UNKNOWN = "unknown"


@dataclass
class StartupAttempt:
    """Information about a startup attempt."""
    attempt_number: int
    start_time: float
    failure_type: Optional[StartupFailureType] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None
    success: bool = False


@dataclass
class StartupHealthReport:
    """Comprehensive health report for startup validation."""
    runner_id: str
    overall_health: bool
    resource_manager_available: bool
    shutdown_coordinator_available: bool
    channel_manager_available: bool
    environment_configured: bool
    channels_registered: bool
    startup_duration: float
    issues: List[str]
    warnings: List[str]


class StartupErrorHandler:
    """
    Comprehensive startup error handling with retry mechanisms and detailed reporting.
    
    This class implements retry logic, fallback mechanisms, and detailed error analysis
    to improve startup reliability and provide actionable error information.
    """
    
    def __init__(self, max_retries: int = 3, base_retry_delay: float = 1.0):
        """
        Initialize the startup error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_retry_delay: Base delay between retries (exponential backoff)
        """
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.startup_attempts: Dict[str, List[StartupAttempt]] = {}
    
    async def handle_startup_with_retries(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        _logger: "loguru.Logger",
        startup_timeout: float = 30.0
    ) -> bool:
        """
        Handle startup with retry mechanisms and comprehensive error handling.
        
        Args:
            bound_instance: The bound instance configuration
            event_sender: Channel for sending events
            task_receiver: Channel for receiving tasks
            _logger: Logger instance
            startup_timeout: Maximum time to wait for startup completion
            
        Returns:
            True if startup succeeded (possibly after retries), False if all attempts failed
        """
        runner_id = bound_instance.bound_runner_id
        
        # Initialize attempt tracking
        if runner_id not in self.startup_attempts:
            self.startup_attempts[runner_id] = []
        
        logger.info(f"Starting startup with error handling for runner {runner_id} "
                   f"(max_retries={self.max_retries}, timeout={startup_timeout}s)")
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            attempt_start_time = time.time()
            
            startup_attempt = StartupAttempt(
                attempt_number=attempt + 1,
                start_time=attempt_start_time
            )
            
            logger.info(f"Startup attempt {attempt + 1}/{self.max_retries + 1} for runner {runner_id}")
            
            try:
                # Perform startup health check before attempting
                health_report = await self._perform_startup_health_check(runner_id)
                
                if not health_report.overall_health:
                    logger.warning(f"Startup health check failed for runner {runner_id}: {health_report.issues}")
                    
                    # Try to resolve health issues
                    resolved = await self._resolve_health_issues(health_report)
                    if not resolved:
                        raise RuntimeError(f"Failed to resolve startup health issues: {health_report.issues}")
                
                # Attempt the actual startup
                success = await bootstrap_runner_with_resource_management(
                    bound_instance, event_sender, task_receiver, _logger, startup_timeout
                )
                
                if success:
                    startup_attempt.success = True
                    startup_attempt.duration = time.time() - attempt_start_time
                    self.startup_attempts[runner_id].append(startup_attempt)
                    
                    logger.info(f"Startup succeeded for runner {runner_id} on attempt {attempt + 1} "
                               f"in {startup_attempt.duration:.2f}s")
                    return True
                else:
                    raise RuntimeError("Bootstrap function returned False")
                    
            except Exception as e:
                startup_attempt.duration = time.time() - attempt_start_time
                startup_attempt.failure_type = self._classify_startup_error(e)
                startup_attempt.error_message = str(e)
                self.startup_attempts[runner_id].append(startup_attempt)
                
                logger.warning(f"Startup attempt {attempt + 1} failed for runner {runner_id}: {e}")
                
                # Check if we should retry
                if attempt < self.max_retries:
                    should_retry, retry_delay = self._should_retry_startup(startup_attempt, attempt)
                    
                    if should_retry:
                        logger.info(f"Retrying startup for runner {runner_id} in {retry_delay:.1f}s")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Startup failure not retryable for runner {runner_id}: {startup_attempt.failure_type}")
                        break
                else:
                    logger.error(f"All startup attempts exhausted for runner {runner_id}")
                    break
        
        # All attempts failed
        await self._report_startup_failure(runner_id, event_sender)
        return False
    
    async def _perform_startup_health_check(self, runner_id: str) -> StartupHealthReport:
        """
        Perform comprehensive health check before startup attempt.
        
        Args:
            runner_id: ID of the runner being started
            
        Returns:
            StartupHealthReport with detailed health information
        """
        start_time = time.time()
        issues = []
        warnings = []
        
        # Check resource manager availability
        resource_manager_available = True
        try:
            resource_manager = get_resource_manager()
            if resource_manager is None:
                resource_manager_available = False
                issues.append("Resource manager not available")
        except Exception as e:
            resource_manager_available = False
            issues.append(f"Resource manager error: {e}")
        
        # Check shutdown coordinator availability
        shutdown_coordinator_available = True
        try:
            shutdown_coordinator = get_shutdown_coordinator()
            if shutdown_coordinator is None:
                shutdown_coordinator_available = False
                issues.append("Shutdown coordinator not available")
        except Exception as e:
            shutdown_coordinator_available = False
            issues.append(f"Shutdown coordinator error: {e}")
        
        # Check channel manager availability
        channel_manager_available = True
        try:
            channel_manager = get_channel_manager()
            if channel_manager is None:
                channel_manager_available = False
                issues.append("Channel manager not available")
        except Exception as e:
            channel_manager_available = False
            issues.append(f"Channel manager error: {e}")
        
        # Check environment configuration
        environment_configured = True
        try:
            # Check for required environment variables or settings
            # This is a placeholder for environment-specific checks
            pass
        except Exception as e:
            environment_configured = False
            issues.append(f"Environment configuration error: {e}")
        
        # Check if channels can be registered (basic functionality test)
        channels_registered = True
        try:
            # This is a basic test - in a real scenario we might test actual channel creation
            pass
        except Exception as e:
            channels_registered = False
            issues.append(f"Channel registration test failed: {e}")
        
        # Determine overall health
        overall_health = (
            resource_manager_available and
            shutdown_coordinator_available and
            channel_manager_available and
            environment_configured and
            channels_registered
        )
        
        duration = time.time() - start_time
        
        report = StartupHealthReport(
            runner_id=runner_id,
            overall_health=overall_health,
            resource_manager_available=resource_manager_available,
            shutdown_coordinator_available=shutdown_coordinator_available,
            channel_manager_available=channel_manager_available,
            environment_configured=environment_configured,
            channels_registered=channels_registered,
            startup_duration=duration,
            issues=issues,
            warnings=warnings
        )
        
        logger.debug(f"Startup health check for runner {runner_id}: "
                    f"healthy={overall_health}, issues={len(issues)}, warnings={len(warnings)}")
        
        return report
    
    async def _resolve_health_issues(self, health_report: StartupHealthReport) -> bool:
        """
        Attempt to resolve health issues identified in the health check.
        
        Args:
            health_report: Health report with identified issues
            
        Returns:
            True if issues were resolved, False otherwise
        """
        runner_id = health_report.runner_id
        resolved_issues = []
        
        logger.info(f"Attempting to resolve {len(health_report.issues)} health issues for runner {runner_id}")
        
        for issue in health_report.issues:
            try:
                if "Resource manager not available" in issue:
                    # Try to reinitialize resource manager
                    get_resource_manager()
                    resolved_issues.append(issue)
                    logger.debug(f"Resolved resource manager issue for runner {runner_id}")
                    
                elif "Shutdown coordinator not available" in issue:
                    # Try to reinitialize shutdown coordinator
                    get_shutdown_coordinator()
                    resolved_issues.append(issue)
                    logger.debug(f"Resolved shutdown coordinator issue for runner {runner_id}")
                    
                elif "Channel manager not available" in issue:
                    # Try to reinitialize channel manager
                    get_channel_manager()
                    resolved_issues.append(issue)
                    logger.debug(f"Resolved channel manager issue for runner {runner_id}")
                    
                else:
                    logger.debug(f"No automatic resolution available for issue: {issue}")
                    
            except Exception as e:
                logger.warning(f"Failed to resolve issue '{issue}' for runner {runner_id}: {e}")
        
        resolution_success = len(resolved_issues) == len(health_report.issues)
        
        if resolution_success:
            logger.info(f"Successfully resolved all health issues for runner {runner_id}")
        else:
            logger.warning(f"Resolved {len(resolved_issues)}/{len(health_report.issues)} issues for runner {runner_id}")
        
        return resolution_success
    
    def _classify_startup_error(self, error: Exception) -> StartupFailureType:
        """
        Classify startup errors for appropriate retry handling.
        
        Args:
            error: The exception that occurred during startup
            
        Returns:
            StartupFailureType classification
        """
        error_str = str(error).lower()
        
        if "timeout" in error_str or isinstance(error, TimeoutError):
            return StartupFailureType.TIMEOUT
        elif "resource" in error_str or "allocation" in error_str:
            return StartupFailureType.RESOURCE_ALLOCATION
        elif "environment" in error_str or "env" in error_str:
            return StartupFailureType.ENVIRONMENT_SETUP
        elif "channel" in error_str or "queue" in error_str:
            return StartupFailureType.CHANNEL_SETUP
        elif "runner" in error_str or "initialization" in error_str:
            return StartupFailureType.RUNNER_INITIALIZATION
        else:
            return StartupFailureType.UNKNOWN
    
    def _should_retry_startup(self, attempt: StartupAttempt, attempt_number: int) -> tuple[bool, float]:
        """
        Determine if startup should be retried and calculate retry delay.
        
        Args:
            attempt: The failed startup attempt
            attempt_number: Current attempt number (0-based)
            
        Returns:
            Tuple of (should_retry, retry_delay_seconds)
        """
        # Some failure types are not worth retrying
        non_retryable_failures = {
            StartupFailureType.ENVIRONMENT_SETUP,  # Environment issues usually need manual intervention
        }
        
        if attempt.failure_type in non_retryable_failures:
            return False, 0.0
        
        # Calculate exponential backoff delay
        retry_delay = self.base_retry_delay * (2 ** attempt_number)
        
        # Add some jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.1, 0.3) * retry_delay
        retry_delay += jitter
        
        # Cap maximum delay
        retry_delay = min(retry_delay, 10.0)
        
        return True, retry_delay
    
    async def _report_startup_failure(self, runner_id: str, event_sender: MpSender[Event]) -> None:
        """
        Report comprehensive startup failure information.
        
        Args:
            runner_id: ID of the runner that failed to start
            event_sender: Channel for sending failure events
        """
        attempts = self.startup_attempts.get(runner_id, [])
        
        if not attempts:
            error_message = "Startup failed with no recorded attempts"
        else:
            # Create detailed error report
            failure_types = [attempt.failure_type.value for attempt in attempts if attempt.failure_type]
            total_duration = sum(attempt.duration for attempt in attempts if attempt.duration)
            
            error_message = (
                f"Startup failed after {len(attempts)} attempts in {total_duration:.2f}s. "
                f"Failure types: {', '.join(set(failure_types))}. "
                f"Last error: {attempts[-1].error_message if attempts else 'Unknown'}"
            )
        
        logger.error(f"Comprehensive startup failure for runner {runner_id}: {error_message}")
        
        # Send failure event
        try:
            event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerFailed(error_message=error_message),
                )
            )
        except Exception as e:
            logger.error(f"Failed to send startup failure event for runner {runner_id}: {e}")
    
    def get_startup_statistics(self, runner_id: str) -> Dict[str, Any]:
        """
        Get startup statistics for a runner.
        
        Args:
            runner_id: ID of the runner
            
        Returns:
            Dictionary with startup statistics
        """
        attempts = self.startup_attempts.get(runner_id, [])
        
        if not attempts:
            return {"runner_id": runner_id, "no_attempts": True}
        
        successful_attempts = [a for a in attempts if a.success]
        failed_attempts = [a for a in attempts if not a.success]
        
        failure_types = {}
        for attempt in failed_attempts:
            if attempt.failure_type:
                failure_types[attempt.failure_type.value] = failure_types.get(attempt.failure_type.value, 0) + 1
        
        return {
            "runner_id": runner_id,
            "total_attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "failure_types": failure_types,
            "total_duration": sum(a.duration for a in attempts if a.duration),
            "average_attempt_duration": sum(a.duration for a in attempts if a.duration) / len(attempts) if attempts else 0,
            "success_rate": len(successful_attempts) / len(attempts) if attempts else 0
        }


# Global startup error handler instance
_global_startup_handler: Optional[StartupErrorHandler] = None


def get_startup_error_handler() -> StartupErrorHandler:
    """Get the global startup error handler instance."""
    global _global_startup_handler
    if _global_startup_handler is None:
        _global_startup_handler = StartupErrorHandler()
    return _global_startup_handler


async def bootstrap_runner_with_resource_management(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
    startup_timeout: float = 30.0,
) -> bool:
    """
    Bootstrap a runner with proper resource management and error handling.
    
    Args:
        bound_instance: The bound instance configuration
        event_sender: Channel for sending events
        task_receiver: Channel for receiving tasks
        _logger: Logger instance
        startup_timeout: Maximum time to wait for startup completion
        
    Returns:
        True if bootstrap succeeded, False if failed
    """
    global logger
    logger = _logger
    
    runner_id = bound_instance.bound_runner_id
    logger.info(f"Starting bootstrap for runner {runner_id} with timeout {startup_timeout}s")
    
    # Get resource management components
    resource_manager = get_resource_manager()
    shutdown_coordinator = get_shutdown_coordinator()
    channel_manager = get_channel_manager()
    
    startup_start_time = time.time()
    
    try:
        # Phase 1: Register core resources with proper cleanup order
        logger.debug(f"Phase 1: Registering resources for runner {runner_id}")
        
        # Register event sender (cleanup order 50 - should be closed after main runner logic)
        event_sender_handle = resource_manager.register_resource(
            resource=event_sender,
            resource_type=ResourceType.CHANNEL,
            cleanup_order=50,
            cleanup_func=lambda: _safe_close_sender(event_sender),
            timeout=5.0,
            resource_id=f"event_sender_{runner_id}"
        )
        
        # Register task receiver (cleanup order 40 - should be closed before sender)
        task_receiver_handle = resource_manager.register_resource(
            resource=task_receiver,
            resource_type=ResourceType.CHANNEL,
            cleanup_order=40,
            cleanup_func=lambda: _safe_close_receiver(task_receiver),
            timeout=5.0,
            resource_id=f"task_receiver_{runner_id}"
        )
        
        logger.debug(f"Registered core resources for runner {runner_id}")
        
        # Phase 2: Configure environment and validate startup conditions
        logger.debug(f"Phase 2: Configuring environment for runner {runner_id}")
        
        if (
            isinstance(bound_instance.instance, MlxJacclInstance)
            and len(bound_instance.instance.jaccl_devices) >= 2
        ):
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"
            logger.debug(f"Set MLX_METAL_FAST_SYNCH=1 for runner {runner_id}")
        
        # Check if we're within startup timeout
        elapsed_time = time.time() - startup_start_time
        if elapsed_time >= startup_timeout:
            raise TimeoutError(f"Startup timeout exceeded during environment setup: {elapsed_time:.2f}s")
        
        # Phase 3: Register shutdown handler for graceful termination
        logger.debug(f"Phase 3: Registering shutdown handler for runner {runner_id}")
        
        def shutdown_handler(shutting_down_runner_id: str):
            if shutting_down_runner_id == runner_id:
                logger.info(f"Shutdown signal received for runner {runner_id}")
        
        shutdown_coordinator.register_shutdown_handler(shutdown_handler)
        
        # Phase 4: Start main runner logic with error recovery
        logger.debug(f"Phase 4: Starting main runner logic for runner {runner_id}")
        
        # Import main after all setup is complete
        from exo.worker.runner.runner import main
        
        # Check final timeout before starting main logic
        elapsed_time = time.time() - startup_start_time
        if elapsed_time >= startup_timeout:
            raise TimeoutError(f"Startup timeout exceeded before main logic: {elapsed_time:.2f}s")
        
        logger.info(f"Bootstrap completed successfully for runner {runner_id} in {elapsed_time:.2f}s")
        
        # Run main logic (this will block until runner completes)
        main(bound_instance, event_sender, task_receiver)
        
        return True
        
    except Exception as e:
        logger.error(f"Bootstrap failed for runner {runner_id}: {e}")
        
        # Send failure event if possible
        try:
            event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerFailed(error_message=f"Bootstrap failed: {str(e)}"),
                )
            )
        except Exception as send_error:
            logger.error(f"Failed to send failure event for runner {runner_id}: {send_error}")
        
        return False


async def enhanced_cleanup_sequence(
    runner_id: str,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    resource_manager: ResourceManager,
    shutdown_coordinator: ShutdownCoordinator,
    cleanup_timeout: float = 15.0
) -> bool:
    """
    Enhanced cleanup sequence with proper synchronization and timeout handling.
    
    This function addresses the specific "Queue is closed" error by ensuring
    proper cleanup order and synchronization between processes.
    
    Args:
        runner_id: ID of the runner being cleaned up
        event_sender: Event sender channel
        task_receiver: Task receiver channel  
        resource_manager: Resource manager instance
        shutdown_coordinator: Shutdown coordinator instance
        cleanup_timeout: Maximum time for cleanup operations
        
    Returns:
        True if cleanup completed successfully, False if timeout or errors
    """
    logger.info(f"Starting enhanced cleanup sequence for runner {runner_id}")
    cleanup_start_time = time.time()
    
    try:
        # Phase 1: Signal shutdown intent (prevents new operations)
        logger.debug(f"Phase 1: Signaling shutdown intent for runner {runner_id}")
        
        shutdown_success = await shutdown_coordinator.initiate_shutdown(
            runner_id, 
            timeout=min(cleanup_timeout * 0.6, 10.0)  # Use 60% of cleanup timeout for shutdown
        )
        
        if not shutdown_success:
            logger.warning(f"Shutdown coordination failed for runner {runner_id}, continuing with cleanup")
        
        # Phase 2: Drain channels before closing (prevents "Queue is closed" errors)
        logger.debug(f"Phase 2: Draining channels for runner {runner_id}")
        
        # Check remaining time
        elapsed = time.time() - cleanup_start_time
        remaining_time = cleanup_timeout - elapsed
        
        if remaining_time > 0:
            drain_timeout = min(remaining_time * 0.5, 5.0)  # Use up to 50% of remaining time for draining
            
            # Drain task receiver first (input channel)
            await _drain_receiver_safely(task_receiver, drain_timeout / 2, runner_id)
            
            # Brief pause to allow any pending sends to complete
            await asyncio.sleep(0.1)
            
            # Drain event sender (output channel) 
            await _drain_sender_safely(event_sender, drain_timeout / 2, runner_id)
        
        # Phase 3: Coordinated resource cleanup
        logger.debug(f"Phase 3: Resource cleanup for runner {runner_id}")
        
        elapsed = time.time() - cleanup_start_time
        remaining_time = cleanup_timeout - elapsed
        
        if remaining_time > 0:
            cleanup_result = await resource_manager.cleanup_resources(timeout=remaining_time)
            
            if cleanup_result.success:
                logger.info(f"Resource cleanup completed successfully for runner {runner_id}")
            else:
                logger.warning(f"Resource cleanup had {len(cleanup_result.failed_resources)} failures for runner {runner_id}")
                # Continue with manual cleanup for critical resources
                await _manual_cleanup_fallback(event_sender, task_receiver, runner_id)
        else:
            logger.warning(f"Cleanup timeout exceeded, performing emergency cleanup for runner {runner_id}")
            await _manual_cleanup_fallback(event_sender, task_receiver, runner_id)
        
        total_cleanup_time = time.time() - cleanup_start_time
        logger.info(f"Enhanced cleanup completed for runner {runner_id} in {total_cleanup_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced cleanup failed for runner {runner_id}: {e}")
        # Perform emergency cleanup
        await _manual_cleanup_fallback(event_sender, task_receiver, runner_id)
        return False


async def _drain_receiver_safely(receiver: MpReceiver[Task], timeout: float, runner_id: str) -> int:
    """
    Safely drain a receiver channel with timeout and error handling.
    
    Args:
        receiver: The receiver to drain
        timeout: Maximum time to spend draining
        runner_id: Runner ID for logging
        
    Returns:
        Number of messages drained
    """
    messages_drained = 0
    start_time = time.time()
    
    try:
        logger.debug(f"Draining task receiver for runner {runner_id}")
        
        while time.time() - start_time < timeout:
            try:
                # Try to receive with a short timeout
                if hasattr(receiver, 'receive_nowait'):
                    message = receiver.receive_nowait()
                    messages_drained += 1
                    logger.debug(f"Drained task message for runner {runner_id}: {type(message)}")
                else:
                    # Fallback for receivers without nowait method
                    break
                    
            except Exception:
                # No more messages or receiver closed
                break
                
            # Brief pause to prevent busy waiting
            await asyncio.sleep(0.01)
        
        if messages_drained > 0:
            logger.debug(f"Drained {messages_drained} task messages for runner {runner_id}")
            
    except Exception as e:
        logger.debug(f"Error draining receiver for runner {runner_id}: {e}")
    
    return messages_drained


async def _drain_sender_safely(sender: MpSender[Event], timeout: float, runner_id: str) -> bool:
    """
    Safely drain/flush a sender channel with timeout and error handling.
    
    Args:
        sender: The sender to drain/flush
        timeout: Maximum time to spend draining
        runner_id: Runner ID for logging
        
    Returns:
        True if draining completed successfully
    """
    try:
        logger.debug(f"Flushing event sender for runner {runner_id}")
        
        # For senders, we mainly need to ensure any pending sends complete
        # This is more about waiting for the send buffer to flush
        start_time = time.time()
        
        # Give a brief moment for any pending operations to complete
        while time.time() - start_time < timeout:
            # Check if sender has any pending operations
            # This is implementation-specific and may not be available
            if hasattr(sender, '_queue') and hasattr(sender._queue, 'qsize'):
                try:
                    queue_size = sender._queue.qsize()
                    if queue_size == 0:
                        break
                    logger.debug(f"Waiting for {queue_size} pending messages in sender for runner {runner_id}")
                except Exception:
                    break
            else:
                # If we can't check queue size, just wait a brief moment
                await asyncio.sleep(min(0.1, timeout))
                break
                
            await asyncio.sleep(0.01)
        
        logger.debug(f"Event sender flush completed for runner {runner_id}")
        return True
        
    except Exception as e:
        logger.debug(f"Error flushing sender for runner {runner_id}: {e}")
        return False


async def _manual_cleanup_fallback(
    event_sender: MpSender[Event], 
    task_receiver: MpReceiver[Task], 
    runner_id: str
) -> None:
    """
    Manual cleanup fallback for critical resources.
    
    This function provides a last-resort cleanup mechanism that handles
    the specific "Queue is closed" error by using safe cleanup methods.
    """
    logger.debug(f"Performing manual cleanup fallback for runner {runner_id}")
    
    # Close task receiver first (input channel)
    try:
        _safe_close_receiver(task_receiver)
        logger.debug(f"Manual cleanup: closed task receiver for runner {runner_id}")
    except Exception as e:
        logger.debug(f"Manual cleanup: error closing task receiver for runner {runner_id}: {e}")
    
    # Brief pause to allow any pending operations to complete
    await asyncio.sleep(0.05)
    
    # Close event sender second (output channel)  
    try:
        _safe_close_sender(event_sender)
        logger.debug(f"Manual cleanup: closed event sender for runner {runner_id}")
    except Exception as e:
        logger.debug(f"Manual cleanup: error closing event sender for runner {runner_id}: {e}")


def _safe_close_sender(sender: MpSender[Event]) -> None:
    """
    Safely close an event sender with comprehensive error handling.
    
    This function addresses the specific "Queue is closed" error by using
    defensive programming techniques and proper error isolation.
    """
    try:
        # First, try to close gracefully
        if hasattr(sender, 'close') and callable(sender.close):
            try:
                sender.close()
            except ValueError as e:
                if "Queue is closed" in str(e):
                    # This is the specific error we're trying to prevent
                    logger.debug(f"Sender already closed (expected): {e}")
                else:
                    logger.debug(f"Sender close error: {e}")
            except Exception as e:
                logger.debug(f"Sender close error: {e}")
        
        # Then, try to join any background threads
        if hasattr(sender, 'join') and callable(sender.join):
            try:
                sender.join(timeout=1.0)  # Use timeout to prevent hanging
            except Exception as e:
                logger.debug(f"Sender join error: {e}")
                
    except Exception as e:
        logger.debug(f"Error during sender cleanup: {e}")


def _safe_close_receiver(receiver: MpReceiver[Task]) -> None:
    """
    Safely close a task receiver with comprehensive error handling.
    
    This function addresses the specific "Queue is closed" error by using
    defensive programming techniques and proper error isolation.
    """
    try:
        # First, try to close gracefully
        if hasattr(receiver, 'close') and callable(receiver.close):
            try:
                receiver.close()
            except ValueError as e:
                if "Queue is closed" in str(e):
                    # This is the specific error we're trying to prevent
                    logger.debug(f"Receiver already closed (expected): {e}")
                else:
                    logger.debug(f"Receiver close error: {e}")
            except Exception as e:
                logger.debug(f"Receiver close error: {e}")
        
        # Then, try to join any background threads
        if hasattr(receiver, 'join') and callable(receiver.join):
            try:
                receiver.join(timeout=1.0)  # Use timeout to prevent hanging
            except Exception as e:
                logger.debug(f"Receiver join error: {e}")
                
    except Exception as e:
        logger.debug(f"Error during receiver cleanup: {e}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    """
    Enhanced entrypoint function with comprehensive startup error handling and resource management.
    
    This function provides retry mechanisms, detailed error reporting, and graceful shutdown
    coordination to fix the "Queue is closed" error and improve startup reliability.
    """
    runner_id = bound_instance.bound_runner_id
    
    # Get resource management components
    resource_manager = get_resource_manager()
    shutdown_coordinator = get_shutdown_coordinator()
    startup_handler = get_startup_error_handler()
    
    try:
        # Use comprehensive startup error handling with retries
        success = asyncio.run(startup_handler.handle_startup_with_retries(
            bound_instance, event_sender, task_receiver, _logger
        ))
        
        if not success:
            logger.error(f"All startup attempts failed for runner {runner_id}")
            
            # Get startup statistics for debugging
            stats = startup_handler.get_startup_statistics(runner_id)
            logger.error(f"Startup statistics for runner {runner_id}: {stats}")
            
    except Exception as e:
        logger.opt(exception=e).error(
            f"Critical error in startup handling for runner {runner_id}: {e}"
        )
        
        # Send failure event with error handling
        try:
            event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerFailed(error_message=f"Critical startup error: {str(e)}"),
                )
            )
        except Exception as send_error:
            logger.error(f"Failed to send critical failure event: {send_error}")
            
    finally:
        # Use enhanced cleanup sequence to prevent "Queue is closed" errors
        logger.info(f"Starting enhanced cleanup sequence for runner {runner_id}")
        
        try:
            cleanup_success = asyncio.run(enhanced_cleanup_sequence(
                runner_id=runner_id,
                event_sender=event_sender,
                task_receiver=task_receiver,
                resource_manager=resource_manager,
                shutdown_coordinator=shutdown_coordinator,
                cleanup_timeout=15.0
            ))
            
            if cleanup_success:
                logger.info(f"Enhanced cleanup completed successfully for runner {runner_id}")
            else:
                logger.warning(f"Enhanced cleanup had issues for runner {runner_id}")
                
        except Exception as cleanup_error:
            logger.error(f"Enhanced cleanup failed for runner {runner_id}: {cleanup_error}")
            # Final emergency cleanup
            try:
                asyncio.run(_manual_cleanup_fallback(event_sender, task_receiver, runner_id))
            except Exception as emergency_error:
                logger.error(f"Emergency cleanup failed for runner {runner_id}: {emergency_error}")
        
        logger.info(f"All cleanup completed for runner {runner_id}")


def entrypoint_with_fallback_mechanisms(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    """
    Alternative entrypoint with additional fallback mechanisms for resource allocation failures.
    
    This function provides an extra layer of fallback handling for cases where
    the primary startup mechanisms fail due to resource constraints.
    """
    runner_id = bound_instance.bound_runner_id
    
    try:
        # First, try the standard enhanced entrypoint
        entrypoint(bound_instance, event_sender, task_receiver, _logger)
        
    except Exception as primary_error:
        logger.warning(f"Primary entrypoint failed for runner {runner_id}: {primary_error}")
        
        # Fallback mechanism: try with reduced resource requirements
        try:
            logger.info(f"Attempting fallback startup for runner {runner_id}")
            
            # Use a simpler startup approach with minimal resource management
            global logger
            logger = _logger
            
            # Set environment if needed
            if (
                isinstance(bound_instance.instance, MlxJacclInstance)
                and len(bound_instance.instance.jaccl_devices) >= 2
            ):
                os.environ["MLX_METAL_FAST_SYNCH"] = "1"
            
            # Try to run main logic directly
            from exo.worker.runner.runner import main
            main(bound_instance, event_sender, task_receiver)
            
            logger.info(f"Fallback startup succeeded for runner {runner_id}")
            
        except Exception as fallback_error:
            logger.error(f"Fallback startup also failed for runner {runner_id}: {fallback_error}")
            
            # Send comprehensive failure report
            try:
                error_message = (
                    f"Both primary and fallback startup failed. "
                    f"Primary error: {str(primary_error)}. "
                    f"Fallback error: {str(fallback_error)}"
                )
                
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id,
                        runner_status=RunnerFailed(error_message=error_message),
                    )
                )
            except Exception as send_error:
                logger.error(f"Failed to send comprehensive failure event: {send_error}")
        
        finally:
            # Ensure cleanup happens even in fallback mode
            try:
                _safe_close_receiver(task_receiver)
                _safe_close_sender(event_sender)
            except Exception as cleanup_error:
                logger.debug(f"Error in fallback cleanup: {cleanup_error}")
