import contextlib
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Process
from typing import Optional, Self, Any

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    create_task_group,
    to_thread,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.shared.types.events import Event, RunnerStatusUpdated, TaskAcknowledged
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerStatus,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender, Sender, mp_channel
from exo.worker.runner.bootstrap import entrypoint
from exo.worker.runner.error_handler import (
    ErrorHandler,
    RecoveryAction,
    get_error_handler,
)
from exo.worker.runner.lifecycle_logger import (
    LifecycleEventType,
    LifecycleLogger,
    get_lifecycle_logger,
)
from exo.worker.runner.resource_manager import (
    ResourceManager,
    ResourceState,
    ResourceType,
    get_resource_manager,
)
from exo.worker.runner.shutdown_coordinator import (
    ShutdownCoordinator,
    get_shutdown_coordinator,
)

PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


@dataclass(eq=False)
class RunnerSupervisor:
    shard_metadata: ShardMetadata
    bound_instance: BoundInstance
    runner_process: Process
    initialize_timeout: float
    _ev_recv: MpReceiver[Event]
    _task_sender: MpSender[Task]
    _event_sender: Sender[Event]
    # err_path: str
    _tg: TaskGroup | None = field(default=None, init=False)
    status: RunnerStatus = field(default_factory=RunnerIdle, init=False)
    pending: dict[TaskId, anyio.Event] = field(default_factory=dict, init=False)

    # Enhanced lifecycle management
    _shutdown_coordinator: ShutdownCoordinator = field(
        default_factory=get_shutdown_coordinator, init=False
    )
    _resource_manager: ResourceManager = field(
        default_factory=get_resource_manager, init=False
    )
    _error_handler: ErrorHandler = field(default_factory=get_error_handler, init=False)
    _lifecycle_logger: LifecycleLogger = field(
        default_factory=get_lifecycle_logger, init=False
    )
    _runner_id: str = field(default="", init=False)
    _resources_registered: bool = field(default=False, init=False)
    _shutdown_in_progress: bool = field(default=False, init=False)
    _health_check_task: Optional[Any] = field(default=None, init=False)

    @classmethod
    def create(
        cls,
        *,
        bound_instance: BoundInstance,
        event_sender: Sender[Event],
        initialize_timeout: float = 400,
    ) -> Self:
        ev_send, ev_recv = mp_channel[Event]()
        # A task is kind of a runner command
        task_sender, task_recv = mp_channel[Task]()

        runner_process = Process(
            target=entrypoint,
            args=(
                bound_instance,
                ev_send,
                task_recv,
                logger,
            ),
            daemon=True,
        )

        shard_metadata = bound_instance.bound_shard

        self = cls(
            bound_instance=bound_instance,
            shard_metadata=shard_metadata,
            runner_process=runner_process,
            initialize_timeout=initialize_timeout,
            _ev_recv=ev_recv,
            _task_sender=task_sender,
            _event_sender=event_sender,
            # err_path=err_path,
        )

        # Generate unique runner ID for coordination
        self._runner_id = f"runner_{bound_instance.bound_runner_id}_{id(self)}"

        # Register shutdown handler for coordination
        self._shutdown_coordinator.register_shutdown_handler(
            self._handle_shutdown_signal
        )

        # Configure error handler retry policies
        self._configure_error_handling()

        logger.debug(f"Created RunnerSupervisor with ID: {self._runner_id}")

        # Log runner creation
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self._lifecycle_logger.log_runner_created(
                    runner_id=self._runner_id,
                    metadata={
                        "bound_runner_id": bound_instance.bound_runner_id,
                        "shard_metadata": str(shard_metadata),
                        "initialize_timeout": initialize_timeout,
                    },
                )
            )
        except RuntimeError:
            # No event loop running, skip async logging for now
            pass

        return self

    def _configure_error_handling(self):
        """Configure error handling policies for this runner."""
        from exo.worker.runner.error_handler import RetryConfig

        # Configure retry policies for different operations
        self._error_handler.register_retry_config(
            "resource_cleanup",
            RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0),
        )

        self._error_handler.register_retry_config(
            "task_send", RetryConfig(max_attempts=2, base_delay=0.5, max_delay=5.0)
        )

        self._error_handler.register_retry_config(
            "event_forward", RetryConfig(max_attempts=2, base_delay=0.1, max_delay=2.0)
        )

        self._error_handler.register_retry_config(
            "process_join",
            RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                max_delay=15.0,
                exponential_backoff=False,
            ),
        )

    async def run(self):
        """Run the runner supervisor with enhanced lifecycle management."""
        correlation_id = self._lifecycle_logger.generate_correlation_id()

        async with self._lifecycle_logger.correlation_context(correlation_id):
            try:
                # Log runner starting
                await self._lifecycle_logger.log_runner_starting(
                    runner_id=self._runner_id,
                    metadata={"correlation_id": correlation_id},
                )

                start_time = time.perf_counter()

                # Register resources with ResourceManager before starting
                await self._register_resources()

                # Start the runner process
                self.runner_process.start()
                logger.info(f"Started runner process for {self._runner_id}")

                # Log successful start
                duration_ms = (time.perf_counter() - start_time) * 1000
                await self._lifecycle_logger.log_runner_started(
                    runner_id=self._runner_id,
                    duration_ms=duration_ms,
                    metadata={
                        "correlation_id": correlation_id,
                        "process_pid": self.runner_process.pid,
                    },
                )

                async with create_task_group() as tg:
                    self._tg = tg
                    tg.start_soon(self._forward_events)

                    # Start health monitoring
                    if self._health_check_task is None:
                        self._health_check_task = tg.start_soon(self._health_monitor)

            except Exception as e:
                logger.error(f"Error starting runner {self._runner_id}: {e}")

                # Log startup failure
                await self._lifecycle_logger.log_event(
                    event_type=LifecycleEventType.RUNNER_FAILED,
                    runner_id=self._runner_id,
                    component="RunnerSupervisor",
                    operation="startup",
                    success=False,
                    error_message=str(e),
                    metadata={"correlation_id": correlation_id},
                )

                # Handle startup failure with error handler
                recovery_action = await self._error_handler.handle_error(
                    error=e,
                    component="RunnerSupervisor",
                    operation="startup",
                    runner_id=self._runner_id,
                    additional_info={"initialize_timeout": self.initialize_timeout},
                )

                await self._handle_startup_failure(e, recovery_action)
                raise
            finally:
                # Ensure graceful shutdown
                await self._graceful_shutdown()

    async def _register_resources(self):
        """Register all resources with the ResourceManager."""
        if self._resources_registered:
            return

        try:
            # Register event receiver
            self._resource_manager.register_resource(
                resource=self._ev_recv,
                resource_type=ResourceType.CHANNEL,
                cleanup_order=10,
                cleanup_func=self._ev_recv.close,
                resource_id=f"{self._runner_id}_ev_recv",
            )

            await self._lifecycle_logger.log_resource_operation(
                runner_id=self._runner_id,
                resource_id=f"{self._runner_id}_ev_recv",
                operation="register",
                metadata={"resource_type": "event_receiver"},
            )

            # Register task sender
            self._resource_manager.register_resource(
                resource=self._task_sender,
                resource_type=ResourceType.CHANNEL,
                cleanup_order=10,
                cleanup_func=self._task_sender.close,
                resource_id=f"{self._runner_id}_task_sender",
            )

            await self._lifecycle_logger.log_resource_operation(
                runner_id=self._runner_id,
                resource_id=f"{self._runner_id}_task_sender",
                operation="register",
                metadata={"resource_type": "task_sender"},
            )

            # Register event sender
            self._resource_manager.register_resource(
                resource=self._event_sender,
                resource_type=ResourceType.CHANNEL,
                cleanup_order=10,
                cleanup_func=self._event_sender.close,
                resource_id=f"{self._runner_id}_event_sender",
            )

            await self._lifecycle_logger.log_resource_operation(
                runner_id=self._runner_id,
                resource_id=f"{self._runner_id}_event_sender",
                operation="register",
                metadata={"resource_type": "event_sender"},
            )

            # Register runner process
            self._resource_manager.register_resource(
                resource=self.runner_process,
                resource_type=ResourceType.PROCESS,
                cleanup_order=5,  # Processes should be cleaned up before channels
                async_cleanup_func=self._cleanup_runner_process,
                resource_id=f"{self._runner_id}_process",
            )

            await self._lifecycle_logger.log_resource_operation(
                runner_id=self._runner_id,
                resource_id=f"{self._runner_id}_process",
                operation="register",
                metadata={
                    "resource_type": "process",
                    "process_pid": self.runner_process.pid,
                },
            )

            self._resources_registered = True
            logger.debug(f"Registered resources for runner {self._runner_id}")

        except Exception as e:
            logger.error(
                f"Failed to register resources for runner {self._runner_id}: {e}"
            )

            # Log resource registration failure
            await self._lifecycle_logger.log_resource_operation(
                runner_id=self._runner_id,
                resource_id="all_resources",
                operation="register",
                success=False,
                error_message=str(e),
            )
            raise

    async def _cleanup_runner_process(self):
        """Clean up the runner process with proper timeout handling and error recovery."""
        if not self.runner_process.is_alive():
            logger.debug(f"Runner process {self._runner_id} already terminated")
            return

        try:
            # Use error handler for retry logic
            await self._error_handler.retry_with_backoff(
                self._attempt_graceful_join, "process_join", timeout=30
            )

            if not self.runner_process.is_alive():
                logger.debug(f"Runner process {self._runner_id} shut down gracefully")
                return

        except Exception as e:
            logger.warning(
                f"Graceful join failed for runner process {self._runner_id}: {e}"
            )

        # Escalate to terminate
        try:
            logger.warning(
                f"Runner process {self._runner_id} didn't shutdown gracefully, terminating"
            )
            self.runner_process.terminate()

            await self._error_handler.retry_with_backoff(
                self._attempt_graceful_join, "process_join", timeout=5
            )

            if not self.runner_process.is_alive():
                logger.debug(
                    f"Runner process {self._runner_id} terminated successfully"
                )
                return

        except Exception as e:
            logger.warning(
                f"Terminate failed for runner process {self._runner_id}: {e}"
            )

        # Final escalation to kill
        try:
            logger.critical(
                f"Runner process {self._runner_id} didn't respond to SIGTERM, killing"
            )
            self.runner_process.kill()

            await self._error_handler.retry_with_backoff(
                self._attempt_graceful_join, "process_join", timeout=5
            )

            if not self.runner_process.is_alive():
                logger.debug(f"Runner process {self._runner_id} killed successfully")
                return

        except Exception as e:
            logger.critical(f"Kill failed for runner process {self._runner_id}: {e}")

        logger.critical(
            f"Runner process {self._runner_id} didn't respond to SIGKILL. System resources may have leaked"
        )

    async def _attempt_graceful_join(self, timeout: float):
        """Attempt to join the runner process with timeout."""
        await to_thread.run_sync(self.runner_process.join, timeout)

    async def _graceful_shutdown(self):
        """Perform graceful shutdown using the three-phase protocol."""
        if self._shutdown_in_progress:
            logger.debug(f"Shutdown already in progress for runner {self._runner_id}")
            return

        self._shutdown_in_progress = True
        logger.info(f"Starting graceful shutdown for runner {self._runner_id}")

        # Log shutdown initiation
        await self._lifecycle_logger.log_shutdown_initiated(
            runner_id=self._runner_id,
            timeout=60.0,
            metadata={"shutdown_reason": "graceful_shutdown"},
        )

        shutdown_start_time = time.perf_counter()

        try:
            # Use ShutdownCoordinator for three-phase shutdown
            success = await self._shutdown_coordinator.initiate_shutdown(
                runner_id=self._runner_id,
                timeout=60.0,  # 60 second timeout for complete shutdown
            )

            shutdown_duration_ms = (time.perf_counter() - shutdown_start_time) * 1000

            if success:
                logger.info(f"Graceful shutdown completed for runner {self._runner_id}")

                # Log successful shutdown
                await self._lifecycle_logger.log_event(
                    event_type=LifecycleEventType.SHUTDOWN_COMPLETED,
                    runner_id=self._runner_id,
                    component="RunnerSupervisor",
                    operation="graceful_shutdown",
                    duration_ms=shutdown_duration_ms,
                    success=True,
                )
            else:
                logger.warning(
                    f"Graceful shutdown failed for runner {self._runner_id}, performing cleanup"
                )

                # Log shutdown failure
                await self._lifecycle_logger.log_event(
                    event_type=LifecycleEventType.SHUTDOWN_FAILED,
                    runner_id=self._runner_id,
                    component="RunnerSupervisor",
                    operation="graceful_shutdown",
                    duration_ms=shutdown_duration_ms,
                    success=False,
                    error_message="Shutdown timeout or coordination failure",
                )

            # Clean up resources through ResourceManager
            cleanup_start_time = time.perf_counter()
            cleanup_result = await self._resource_manager.cleanup_resources(
                timeout=30.0
            )
            cleanup_duration_ms = (time.perf_counter() - cleanup_start_time) * 1000

            if cleanup_result.success:
                logger.debug(f"Resource cleanup completed for runner {self._runner_id}")

                # Log successful cleanup
                await self._lifecycle_logger.log_resource_operation(
                    runner_id=self._runner_id,
                    resource_id="all_resources",
                    operation="cleanup_complete",
                    duration_ms=cleanup_duration_ms,
                    success=True,
                )
            else:
                logger.error(
                    f"Resource cleanup failed for runner {self._runner_id}: {cleanup_result.errors}"
                )

                # Log cleanup failure
                await self._lifecycle_logger.log_resource_operation(
                    runner_id=self._runner_id,
                    resource_id="all_resources",
                    operation="cleanup_failed",
                    duration_ms=cleanup_duration_ms,
                    success=False,
                    error_message=str(cleanup_result.errors),
                )

        except Exception as e:
            shutdown_duration_ms = (time.perf_counter() - shutdown_start_time) * 1000
            logger.error(
                f"Error during graceful shutdown of runner {self._runner_id}: {e}"
            )

            # Log shutdown error
            await self._lifecycle_logger.log_event(
                event_type=LifecycleEventType.SHUTDOWN_FAILED,
                runner_id=self._runner_id,
                component="RunnerSupervisor",
                operation="graceful_shutdown",
                duration_ms=shutdown_duration_ms,
                success=False,
                error_message=str(e),
            )

            # Handle shutdown error
            recovery_action = await self._error_handler.handle_error(
                error=e,
                component="RunnerSupervisor",
                operation="graceful_shutdown",
                runner_id=self._runner_id,
            )

            # Log error handling
            await self._lifecycle_logger.log_error_handling(
                runner_id=self._runner_id,
                error=e,
                component="RunnerSupervisor",
                operation="graceful_shutdown",
                recovery_action=recovery_action.value
                if hasattr(recovery_action, "value")
                else str(recovery_action),
            )

            if recovery_action == RecoveryAction.FORCE_CLEANUP:
                logger.warning(f"Forcing cleanup for runner {self._runner_id}")
                try:
                    cleanup_result = await self._resource_manager.cleanup_resources(
                        timeout=10.0
                    )
                    if not cleanup_result.success:
                        logger.error(
                            f"Force cleanup failed for runner {self._runner_id}: {cleanup_result.errors}"
                        )
                except Exception as cleanup_error:
                    logger.error(
                        f"Force cleanup error for runner {self._runner_id}: {cleanup_error}"
                    )

        finally:
            self._shutdown_in_progress = False

    def _handle_shutdown_signal(self, runner_id: str):
        """Handle shutdown signal from ShutdownCoordinator."""
        if runner_id == self._runner_id:
            logger.debug(f"Received shutdown signal for runner {self._runner_id}")
            if self._tg:
                self._tg.cancel_scope.cancel()

    async def _handle_startup_failure(
        self, error: Exception, recovery_action: RecoveryAction
    ):
        """Handle failures during runner startup with enhanced error recovery."""
        logger.error(f"Runner {self._runner_id} startup failed: {error}")

        error_message = f"Startup failed: {error}"

        # Determine if we should attempt recovery based on recovery action
        if recovery_action == RecoveryAction.RETRY:
            logger.info(f"Attempting startup recovery for runner {self._runner_id}")
            # This would be implemented in a full recovery system
            # For now, we just log the intent

        elif recovery_action == RecoveryAction.RESTART_COMPONENT:
            logger.info(f"Component restart recommended for runner {self._runner_id}")
            error_message = f"Startup failed, restart recommended: {error}"

        # Send failure event
        try:
            await self._event_sender.send(
                RunnerStatusUpdated(
                    runner_id=self.bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(error_message=error_message),
                )
            )
        except Exception as send_error:
            # Handle event sending failure
            await self._error_handler.handle_error(
                error=send_error,
                component="RunnerSupervisor",
                operation="send_failure_event",
                runner_id=self._runner_id,
            )

        # Attempt cleanup
        try:
            await self._graceful_shutdown()
        except Exception as cleanup_error:
            logger.error(f"Error during startup failure cleanup: {cleanup_error}")
            await self._error_handler.handle_error(
                error=cleanup_error,
                component="RunnerSupervisor",
                operation="startup_failure_cleanup",
                runner_id=self._runner_id,
            )

    async def _health_monitor(self):
        """Monitor runner health and perform recovery if needed."""
        health_check_interval = 30.0  # Check every 30 seconds
        consecutive_failures = 0
        max_consecutive_failures = 3

        logger.debug(f"Starting health monitoring for runner {self._runner_id}")

        while True:
            try:
                await anyio.sleep(health_check_interval)

                # Check if shutdown is in progress
                if self._shutdown_in_progress:
                    logger.debug(
                        f"Stopping health monitoring due to shutdown for runner {self._runner_id}"
                    )
                    break

                # Perform health check
                health_check_start_time = time.perf_counter()
                health_status = await self._perform_health_check()
                health_check_duration_ms = (
                    time.perf_counter() - health_check_start_time
                ) * 1000

                # Log health check result
                await self._lifecycle_logger.log_health_check(
                    runner_id=self._runner_id,
                    success=health_status["healthy"],
                    duration_ms=health_check_duration_ms,
                    health_score=health_status.get("metrics", {}).get(
                        "health_score", 0.0
                    ),
                    issues=health_status["issues"],
                    metadata=health_status.get("metrics", {}),
                )

                if health_status["healthy"]:
                    # Reset failure counter on successful health check
                    if consecutive_failures > 0:
                        logger.info(
                            f"Runner {self._runner_id} health recovered after {consecutive_failures} failures"
                        )
                        consecutive_failures = 0

                    logger.debug(f"Health check passed for runner {self._runner_id}")
                else:
                    consecutive_failures += 1
                    logger.warning(
                        f"Health check failed for runner {self._runner_id} "
                        f"({consecutive_failures}/{max_consecutive_failures}): {health_status['issues']}"
                    )

                    # Attempt recovery if we haven't exceeded max failures
                    if consecutive_failures < max_consecutive_failures:
                        recovery_success = await self._attempt_health_recovery(
                            health_status
                        )
                        if recovery_success:
                            logger.info(
                                f"Health recovery successful for runner {self._runner_id}"
                            )
                            consecutive_failures = 0
                        else:
                            logger.warning(
                                f"Health recovery failed for runner {self._runner_id}"
                            )
                    else:
                        logger.error(
                            f"Runner {self._runner_id} failed {max_consecutive_failures} consecutive health checks, initiating shutdown"
                        )
                        await self._handle_persistent_failure()
                        break

                # Check for shutdown signals
                shutdown_phase = self._shutdown_coordinator.check_shutdown_signal(
                    self._runner_id
                )
                if shutdown_phase:
                    logger.info(
                        f"Detected shutdown signal for runner {self._runner_id}: {shutdown_phase}"
                    )
                    if self._tg:
                        self._tg.cancel_scope.cancel()
                    break

            except Exception as e:
                logger.error(
                    f"Error in health monitor for runner {self._runner_id}: {e}"
                )

                # Handle health monitoring errors
                recovery_action = await self._error_handler.handle_error(
                    error=e,
                    component="RunnerSupervisor",
                    operation="health_monitoring",
                    runner_id=self._runner_id,
                )

                if recovery_action == RecoveryAction.ESCALATE:
                    logger.error(
                        f"Health monitoring failed critically for runner {self._runner_id}"
                    )
                    break

                # Continue monitoring despite errors
                continue

        logger.debug(f"Health monitoring stopped for runner {self._runner_id}")

    async def _perform_health_check(self) -> dict:
        """Perform comprehensive health check on the runner."""
        health_status = {
            "healthy": True,
            "issues": [],
            "metrics": {},
            "timestamp": datetime.now(),
        }

        try:
            # Check process health
            if not self.runner_process.is_alive():
                health_status["healthy"] = False
                health_status["issues"].append("Process not alive")
            else:
                health_status["metrics"]["process_pid"] = self.runner_process.pid
                health_status["metrics"]["process_alive"] = True

            # Check resource states
            resource_states = self._resource_manager.get_resource_count_by_state()
            error_resources = resource_states.get(ResourceState.ERROR, 0)
            if error_resources > 0:
                health_status["healthy"] = False
                health_status["issues"].append(
                    f"{error_resources} resources in error state"
                )

            # Convert ResourceState enums to strings for serialization
            health_status["metrics"]["resource_states"] = {
                state.name if hasattr(state, "name") else str(state): count
                for state, count in resource_states.items()
            }

            # Check pending tasks (too many might indicate a problem)
            pending_count = len(self.pending)
            if pending_count > 10:  # Configurable threshold
                health_status["healthy"] = False
                health_status["issues"].append(
                    f"Too many pending tasks: {pending_count}"
                )

            health_status["metrics"]["pending_tasks"] = pending_count

            # Check runner status
            if isinstance(self.status, RunnerFailed):
                health_status["healthy"] = False
                health_status["issues"].append(
                    f"Runner status is failed: {self.status}"
                )

            health_status["metrics"]["runner_status"] = type(self.status).__name__

            # Check error handler statistics
            error_stats = self._error_handler.get_error_statistics()
            recent_errors = len(self._error_handler.get_recent_errors(hours=1))

            if recent_errors > 5:  # Configurable threshold
                health_status["healthy"] = False
                health_status["issues"].append(
                    f"Too many recent errors: {recent_errors}"
                )

            health_status["metrics"]["error_statistics"] = error_stats
            health_status["metrics"]["recent_errors"] = recent_errors

            # Performance metrics
            health_status["metrics"]["initialize_timeout"] = self.initialize_timeout
            health_status["metrics"]["shutdown_in_progress"] = (
                self._shutdown_in_progress
            )
            health_status["metrics"]["resources_registered"] = (
                self._resources_registered
            )

            # Calculate health score
            health_score = self._calculate_health_score(health_status["metrics"])
            health_status["metrics"]["health_score"] = health_score

            # Update overall health based on score
            if health_score < 0.7:  # 70% threshold
                health_status["healthy"] = False
                if "Low health score" not in health_status["issues"]:
                    health_status["issues"].append(
                        f"Low health score: {health_score:.2f}"
                    )

        except Exception as e:
            logger.error(
                f"Error performing health check for runner {self._runner_id}: {e}"
            )
            health_status["healthy"] = False
            health_status["issues"].append(f"Health check error: {e}")

        return health_status

    async def _attempt_health_recovery(self, health_status: dict) -> bool:
        """Attempt to recover from health issues."""
        logger.info(f"Attempting health recovery for runner {self._runner_id}")

        recovery_success = True

        try:
            # Recovery strategies based on specific issues
            for issue in health_status["issues"]:
                if "Process not alive" in issue:
                    logger.warning(
                        f"Cannot recover from dead process for runner {self._runner_id}"
                    )
                    recovery_success = False

                elif "resources in error state" in issue:
                    logger.info(
                        f"Attempting resource recovery for runner {self._runner_id}"
                    )
                    # Try to clean up error resources
                    try:
                        cleanup_result = await self._resource_manager.cleanup_resources(
                            timeout=10.0
                        )
                        if not cleanup_result.success:
                            logger.warning(
                                f"Resource cleanup during recovery failed: {cleanup_result.errors}"
                            )
                            recovery_success = False
                    except Exception as e:
                        logger.error(f"Resource recovery failed: {e}")
                        recovery_success = False

                elif "Too many pending tasks" in issue:
                    logger.info(
                        f"Clearing stale pending tasks for runner {self._runner_id}"
                    )
                    # Clear old pending tasks (this is a simplified approach)
                    stale_tasks = []
                    for task_id, event in self.pending.items():
                        if not event.is_set():
                            stale_tasks.append(task_id)

                    for task_id in stale_tasks[:5]:  # Clear up to 5 stale tasks
                        event = self.pending.pop(task_id, None)
                        if event:
                            event.set()
                            logger.debug(f"Cleared stale pending task {task_id}")

                elif "Too many recent errors" in issue:
                    logger.info(
                        f"Clearing old error records for runner {self._runner_id}"
                    )
                    self._error_handler.clear_old_errors(hours=1)

            # Additional recovery actions
            if recovery_success:
                logger.info(
                    f"Health recovery completed successfully for runner {self._runner_id}"
                )
            else:
                logger.warning(
                    f"Health recovery partially failed for runner {self._runner_id}"
                )

        except Exception as e:
            logger.error(
                f"Error during health recovery for runner {self._runner_id}: {e}"
            )
            recovery_success = False

        return recovery_success

    async def _handle_persistent_failure(self):
        """Handle persistent health failures."""
        logger.error(f"Handling persistent failure for runner {self._runner_id}")

        try:
            # Send failure status
            await self._event_sender.send(
                RunnerStatusUpdated(
                    runner_id=self.bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(
                        error_message="Persistent health check failures"
                    ),
                )
            )
        except Exception as e:
            logger.error(f"Failed to send persistent failure status: {e}")

        # Initiate shutdown
        if self._tg:
            self._tg.cancel_scope.cancel()

    def shutdown(self):
        """Initiate shutdown of the runner supervisor."""
        logger.info(f"Shutdown requested for runner {self._runner_id}")

        if self._tg:
            self._tg.cancel_scope.cancel()

        # The actual graceful shutdown will be handled in the run() method's finally block

    async def start_task(self, task: Task):
        """Start a task with enhanced error handling."""
        logger.info(f"Starting task {task} on runner {self._runner_id}")

        # Check if shutdown is in progress
        if self._shutdown_in_progress:
            logger.warning(
                f"Cannot start task {task}, shutdown in progress for runner {self._runner_id}"
            )
            return

        event = anyio.Event()
        self.pending[task.task_id] = event

        try:
            self._task_sender.send(task)
            logger.debug(f"Task {task} sent to runner {self._runner_id}")
        except ClosedResourceError as e:
            # Handle closed resource error specifically
            recovery_action = await self._error_handler.handle_error(
                error=e,
                component="RunnerSupervisor",
                operation="task_send",
                runner_id=self._runner_id,
                additional_info={"task_id": task.task_id},
            )

            if recovery_action == RecoveryAction.SKIP:
                logger.info(
                    f"Skipping task {task} due to closed resource (expected during shutdown)"
                )
            else:
                logger.warning(
                    f"Task {task} dropped, runner {self._runner_id} closed communication."
                )

            self.pending.pop(task.task_id, None)
            return

        except Exception as e:
            # Handle other send errors
            recovery_action = await self._error_handler.handle_error(
                error=e,
                component="RunnerSupervisor",
                operation="task_send",
                runner_id=self._runner_id,
                additional_info={"task_id": task.task_id},
            )

            if recovery_action == RecoveryAction.RETRY:
                # Implement retry logic here if needed
                logger.info(f"Retry recommended for task {task} send failure")

            logger.error(f"Error sending task {task} to runner {self._runner_id}: {e}")
            self.pending.pop(task.task_id, None)
            return

        try:
            await event.wait()
            logger.info(f"Finished task {task} on runner {self._runner_id}")
        except Exception as e:
            logger.error(
                f"Error waiting for task {task} completion on runner {self._runner_id}: {e}"
            )
            self.pending.pop(task.task_id, None)

    async def _forward_events(self):
        """Forward events with enhanced error handling and recovery."""
        logger.debug(f"Starting event forwarding for runner {self._runner_id}")

        try:
            with self._ev_recv as events:
                while not self._shutdown_in_progress:
                    try:
                        # Use explicit receive with proper exception handling for race conditions
                        event = await events.receive_async()

                        # Double-check shutdown status after receiving event
                        if self._shutdown_in_progress:
                            logger.debug(
                                f"Stopping event forwarding due to shutdown for runner {self._runner_id}"
                            )
                            break

                        if isinstance(event, RunnerStatusUpdated):
                            self.status = event.runner_status
                            logger.debug(
                                f"Runner {self._runner_id} status updated: {event.runner_status}"
                            )

                        if isinstance(event, TaskAcknowledged):
                            task_event = self.pending.pop(event.task_id, None)
                            if task_event:
                                task_event.set()
                            logger.debug(
                                f"Task {event.task_id} acknowledged by runner {self._runner_id}"
                            )
                            continue

                        try:
                            await self._event_sender.send(event)
                        except Exception as e:
                            # Handle event forwarding errors
                            recovery_action = await self._error_handler.handle_error(
                                error=e,
                                component="RunnerSupervisor",
                                operation="event_forward",
                                runner_id=self._runner_id,
                                additional_info={"event_type": type(event).__name__},
                            )

                            if recovery_action == RecoveryAction.RETRY:
                                # Implement retry logic for event forwarding
                                try:
                                    await self._error_handler.retry_with_backoff(
                                        self._event_sender.send, "event_forward", event
                                    )
                                except Exception as retry_error:
                                    logger.error(
                                        f"Event forwarding retry failed for runner {self._runner_id}: {retry_error}"
                                    )
                            else:
                                logger.error(
                                    f"Error forwarding event from runner {self._runner_id}: {e}"
                                )
                            # Continue processing other events

                    except (ClosedResourceError, BrokenResourceError) as e:
                        logger.info(
                            f"Event channel closed for runner {self._runner_id}, checking runner status"
                        )

                        # Handle resource errors with error handler
                        recovery_action = await self._error_handler.handle_error(
                            error=e,
                            component="RunnerSupervisor",
                            operation="event_forwarding",
                            runner_id=self._runner_id,
                        )

                        await self._check_runner(e)

                        # Set all pending task events to avoid hanging
                        for task_id, task_event in self.pending.items():
                            task_event.set()
                            logger.debug(
                                f"Released pending task {task_id} due to channel closure"
                            )

                        # Break out of the event loop when channel is closed
                        break

                    except EndOfStream:
                        logger.debug(
                            f"End of stream reached for runner {self._runner_id}, stopping event forwarding"
                        )
                        # Set all pending task events
                        for task_id, task_event in self.pending.items():
                            task_event.set()
                            logger.debug(
                                f"Released pending task {task_id} due to end of stream"
                            )
                        break

        except (ClosedResourceError, BrokenResourceError) as e:
            logger.info(f"Event channel closed for runner {self._runner_id}: {e}")
            # Set all pending task events to avoid hanging
            for task_id, task_event in self.pending.items():
                task_event.set()
                logger.debug(f"Released pending task {task_id} due to channel closure")
        except Exception as e:
            logger.error(
                f"Unexpected error in event forwarding for runner {self._runner_id}: {e}"
            )
            # Set all pending task events
            for task_id, task_event in self.pending.items():
                task_event.set()

        logger.debug(f"Event forwarding stopped for runner {self._runner_id}")

    def __del__(self) -> None:
        """Cleanup when RunnerSupervisor is garbage collected."""
        if self.runner_process.is_alive():
            logger.warning(
                f"RunnerSupervisor {self._runner_id} was not stopped cleanly."
            )
            with contextlib.suppress(ValueError):
                self.runner_process.kill()

    async def _check_runner(self, e: Exception) -> None:
        """Check runner status with enhanced error reporting and recovery."""
        logger.info(f"Checking runner {self._runner_id} status due to: {e}")

        if self.runner_process.is_alive():
            logger.info(
                f"Runner {self._runner_id} was found to be alive, attempting to join process"
            )
            try:
                await to_thread.run_sync(self.runner_process.join, 1)
            except Exception as join_error:
                logger.warning(
                    f"Error joining runner process {self._runner_id}: {join_error}"
                )

        rc = self.runner_process.exitcode
        logger.info(f"Runner {self._runner_id} exited with exit code {rc}")

        if rc == 0:
            logger.info(f"Runner {self._runner_id} exited normally")
            return

        # Determine cause of termination
        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"
        else:
            cause = f"exitcode={rc}"

        error_message = f"Runner {self._runner_id} terminated ({cause})"
        logger.opt(exception=e).error(error_message)

        # Send failure status update
        try:
            await self._event_sender.send(
                RunnerStatusUpdated(
                    runner_id=self.bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(error_message=error_message),
                )
            )
        except Exception as send_error:
            logger.error(
                f"Failed to send failure status for runner {self._runner_id}: {send_error}"
            )

        # Initiate shutdown
        self.shutdown()

    # Additional methods for enhanced functionality

    def get_runner_health(self) -> dict:
        """Get comprehensive health information for the runner."""
        try:
            # Get basic health info
            basic_health = {
                "runner_id": self._runner_id,
                "process_alive": self.runner_process.is_alive(),
                "process_pid": self.runner_process.pid
                if self.runner_process.is_alive()
                else None,
                "status": type(self.status).__name__,
                "pending_tasks": len(self.pending),
                "shutdown_in_progress": self._shutdown_in_progress,
                "resources_registered": self._resources_registered,
            }

            # Add resource manager health
            resource_states = self._resource_manager.get_resource_count_by_state()
            basic_health["resource_states"] = {
                state.value: count for state, count in resource_states.items()
            }

            # Add error handler statistics
            error_stats = self._error_handler.get_error_statistics()
            basic_health["error_statistics"] = error_stats

            # Add recent error count
            recent_errors = len(self._error_handler.get_recent_errors(hours=1))
            basic_health["recent_errors_1h"] = recent_errors

            # Add shutdown coordinator status
            shutdown_status = self._shutdown_coordinator.get_shutdown_status(
                self._runner_id
            )
            if shutdown_status:
                basic_health["shutdown_status"] = {
                    "phase": shutdown_status.phase.value,
                    "started_at": shutdown_status.started_at.isoformat(),
                    "timeout_at": shutdown_status.timeout_at.isoformat(),
                    "error_count": shutdown_status.error_count,
                }

            # Calculate overall health score
            health_score = self._calculate_health_score(basic_health)
            basic_health["health_score"] = health_score
            basic_health["healthy"] = health_score > 0.7  # 70% threshold

            return basic_health

        except Exception as e:
            logger.error(f"Error getting health info for runner {self._runner_id}: {e}")
            return {
                "runner_id": self._runner_id,
                "healthy": False,
                "error": str(e),
                "health_score": 0.0,
            }

    def _calculate_health_score(self, health_info: dict) -> float:
        """Calculate a health score between 0.0 and 1.0."""
        score = 1.0

        # Process health (40% weight)
        if not health_info.get("process_alive", False):
            score -= 0.4

        # Status health (20% weight)
        if "Failed" in health_info.get("status", ""):
            score -= 0.2

        # Resource health (20% weight)
        resource_states = health_info.get("resource_states", {})
        error_resources = resource_states.get("error", 0)
        total_resources = sum(resource_states.values())
        if total_resources > 0:
            error_ratio = error_resources / total_resources
            score -= 0.2 * error_ratio

        # Error health (10% weight)
        error_stats = health_info.get("error_statistics", {})
        success_rate = error_stats.get("success_rate", 1.0)
        score -= 0.1 * (1.0 - success_rate)

        # Recent errors (10% weight)
        recent_errors = health_info.get("recent_errors_1h", 0)
        if recent_errors > 0:
            error_penalty = min(0.1, recent_errors * 0.02)  # 2% per error, max 10%
            score -= error_penalty

        return max(0.0, score)

    async def recover_runner(self) -> bool:
        """Attempt to recover a failed runner with enhanced recovery strategies."""
        logger.info(f"Attempting recovery for runner {self._runner_id}")

        if self.runner_process.is_alive():
            logger.info(
                f"Runner {self._runner_id} is still alive, performing health recovery"
            )
            health_status = await self._perform_health_check()
            return await self._attempt_health_recovery(health_status)

        try:
            # Enhanced recovery for dead processes
            logger.warning(
                f"Runner process {self._runner_id} is dead, attempting full recovery"
            )

            # Step 1: Clean up old resources
            logger.debug(f"Cleaning up old resources for runner {self._runner_id}")
            try:
                cleanup_result = await self._resource_manager.cleanup_resources(
                    timeout=15.0
                )
                if not cleanup_result.success:
                    logger.warning(
                        f"Resource cleanup during recovery had issues: {cleanup_result.errors}"
                    )
            except Exception as e:
                logger.error(f"Resource cleanup failed during recovery: {e}")
                return False

            # Step 2: Reset internal state
            logger.debug(f"Resetting internal state for runner {self._runner_id}")
            self._resources_registered = False
            self._shutdown_in_progress = False
            self.status = RunnerIdle()

            # Clear pending tasks
            for _task_id, event in self.pending.items():
                event.set()
            self.pending.clear()

            # Step 3: Create new process (simplified - in full implementation this would be more complex)
            logger.debug(f"Creating new process for runner {self._runner_id}")

            # Create new channels
            ev_send, ev_recv = mp_channel[Event]()
            task_sender, task_recv = mp_channel[Task]()

            # Create new process
            new_process = Process(
                target=entrypoint,
                args=(
                    self.bound_instance,
                    ev_send,
                    task_recv,
                    logger,
                ),
                daemon=True,
            )

            # Update references
            old_process = self.runner_process
            self.runner_process = new_process
            self._ev_recv = ev_recv
            self._task_sender = task_sender

            # Step 4: Start new process
            logger.debug(f"Starting new process for runner {self._runner_id}")
            self.runner_process.start()

            # Step 5: Re-register resources
            await self._register_resources()

            # Step 6: Clean up old process
            try:
                if old_process.is_alive():
                    old_process.terminate()
                    await to_thread.run_sync(old_process.join, 5)
                    if old_process.is_alive():
                        old_process.kill()
            except Exception as e:
                logger.warning(f"Error cleaning up old process during recovery: {e}")

            logger.info(f"Runner recovery completed successfully for {self._runner_id}")
            return True

        except Exception as e:
            logger.error(f"Error during runner recovery for {self._runner_id}: {e}")

            # Handle recovery failure
            await self._error_handler.handle_error(
                error=e,
                component="RunnerSupervisor",
                operation="runner_recovery",
                runner_id=self._runner_id,
            )

            return False

    def is_healthy(self) -> bool:
        """Check if the runner is currently healthy."""
        return (
            self.runner_process.is_alive()
            and not self._shutdown_in_progress
            and self._resources_registered
            and not isinstance(self.status, RunnerFailed)
        )
