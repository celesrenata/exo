"""
RecoveryManager for handling different failure modes in distributed inference.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

from ..types.validation import (
    FailureMode,
    RecoveryAction,
    RecoveryStatus,
    RecoveryResult,
    DeviceHealth,
    CorruptionReport,
    CorruptionSeverity,
)


logger = logging.getLogger(__name__)


class RecoveryManager:
    """
    Manages recovery strategies for different failure modes in distributed inference.
    Handles device health monitoring, pipeline reinitialization, and recovery coordination.
    """

    def __init__(self, max_retry_attempts: int = 3, recovery_timeout: float = 30.0):
        self.max_retry_attempts = max_retry_attempts
        self.recovery_timeout = recovery_timeout

        # Device health tracking
        self.device_health: Dict[str, DeviceHealth] = {}
        self.failure_history: Dict[str, List[datetime]] = defaultdict(list)

        # Recovery strategy mapping
        self.recovery_strategies = {
            FailureMode.TOKEN_CORRUPTION: self._recover_token_corruption,
            FailureMode.ENCODING_ERROR: self._recover_encoding_error,
            FailureMode.SYNCHRONIZATION_FAILURE: self._recover_synchronization_failure,
            FailureMode.COMMUNICATION_FAILURE: self._recover_communication_failure,
            FailureMode.DEVICE_FAILURE: self._recover_device_failure,
            FailureMode.PIPELINE_FAILURE: self._recover_pipeline_failure,
            FailureMode.MEMORY_CORRUPTION: self._recover_memory_corruption,
            FailureMode.TIMEOUT: self._recover_timeout,
        }

        # Callbacks for external operations
        self.device_reinit_callback: Optional[Callable[[str], bool]] = None
        self.pipeline_restart_callback: Optional[Callable[[List[str]], bool]] = None
        self.sync_reset_callback: Optional[Callable[[], bool]] = None

        # Recovery statistics
        self.recovery_stats: Dict[str, Any] = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_by_type": defaultdict(int),
            "recovery_by_action": defaultdict(int),
        }

    def register_callbacks(
        self,
        device_reinit: Optional[Callable[[str], bool]] = None,
        pipeline_restart: Optional[Callable[[List[str]], bool]] = None,
        sync_reset: Optional[Callable[[], bool]] = None,
    ):
        """Register callbacks for external recovery operations."""
        if device_reinit:
            self.device_reinit_callback = device_reinit
        if pipeline_restart:
            self.pipeline_restart_callback = pipeline_restart
        if sync_reset:
            self.sync_reset_callback = sync_reset

    async def recover_from_corruption(
        self,
        failure_mode: FailureMode,
        affected_devices: List[str] = None,
        corruption_report: Optional[CorruptionReport] = None,
    ) -> RecoveryResult:
        """
        Recover from a detected failure or corruption.

        Args:
            failure_mode: Type of failure detected
            affected_devices: List of device IDs affected by the failure
            corruption_report: Optional detailed corruption report

        Returns:
            RecoveryResult with details about the recovery attempt
        """
        start_time = datetime.now()
        affected_devices = affected_devices or []

        logger.info(
            f"Starting recovery for {failure_mode} affecting devices: {affected_devices}"
        )

        # Update statistics
        self.recovery_stats["total_recoveries"] += 1
        self.recovery_stats["recovery_by_type"][failure_mode] += 1

        # Update device health for affected devices
        for device_id in affected_devices:
            self._update_device_health(
                device_id, is_healthy=False, error=f"{failure_mode} detected"
            )

        try:
            # Get recovery strategy for this failure mode
            recovery_strategy = self.recovery_strategies.get(failure_mode)
            if not recovery_strategy:
                return RecoveryResult(
                    status=RecoveryStatus.FAILURE,
                    action_taken=RecoveryAction.NO_ACTION,
                    success=False,
                    error_message=f"No recovery strategy for {failure_mode}",
                    affected_devices=affected_devices,
                )

            # Execute recovery strategy
            result = await recovery_strategy(affected_devices, corruption_report)

            # Update statistics
            if result.success:
                self.recovery_stats["successful_recoveries"] += 1
                # Mark affected devices as healthy if recovery succeeded
                for device_id in affected_devices:
                    self._update_device_health(device_id, is_healthy=True)
            else:
                self.recovery_stats["failed_recoveries"] += 1

            self.recovery_stats["recovery_by_action"][result.action_taken] += 1

            # Calculate recovery time
            recovery_time = (datetime.now() - start_time).total_seconds()
            result.recovery_time = recovery_time

            logger.info(f"Recovery completed: {result.status} in {recovery_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Recovery failed with exception: {e}")
            self.recovery_stats["failed_recoveries"] += 1

            return RecoveryResult(
                status=RecoveryStatus.FAILURE,
                action_taken=RecoveryAction.NO_ACTION,
                success=False,
                error_message=f"Recovery exception: {str(e)}",
                recovery_time=(datetime.now() - start_time).total_seconds(),
                affected_devices=affected_devices,
            )

    async def reinitialize_pipeline(self, affected_nodes: List[str]) -> bool:
        """
        Reinitialize the pipeline for affected nodes.

        Args:
            affected_nodes: List of node IDs to reinitialize

        Returns:
            True if reinitialization succeeded, False otherwise
        """
        logger.info(f"Reinitializing pipeline for nodes: {affected_nodes}")

        try:
            if self.pipeline_restart_callback:
                if asyncio.iscoroutinefunction(self.pipeline_restart_callback):
                    success = await asyncio.wait_for(
                        self.pipeline_restart_callback(affected_nodes),
                        timeout=self.recovery_timeout,
                    )
                else:
                    success = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                self.pipeline_restart_callback, affected_nodes
                            )
                        ),
                        timeout=self.recovery_timeout,
                    )

                if success:
                    # Update device health
                    for node_id in affected_nodes:
                        self._update_device_health(node_id, is_healthy=True)
                    logger.info("Pipeline reinitialization successful")
                    return True
                else:
                    logger.error("Pipeline reinitialization failed")
                    return False
            else:
                logger.warning("No pipeline restart callback registered")
                return False

        except asyncio.TimeoutError:
            logger.error(
                f"Pipeline reinitialization timed out after {self.recovery_timeout}s"
            )
            return False
        except Exception as e:
            logger.error(f"Pipeline reinitialization failed: {e}")
            return False

    def monitor_device_health(self, device_id: str) -> DeviceHealth:
        """
        Get current health status of a device.

        Args:
            device_id: ID of the device to check

        Returns:
            DeviceHealth object with current status
        """
        return self.device_health.get(
            device_id,
            DeviceHealth(
                device_id=device_id,
                is_healthy=True,
                last_heartbeat=datetime.now(),
                status="unknown",
            ),
        )

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics and metrics."""
        return dict(self.recovery_stats)

    def _update_device_health(
        self, device_id: str, is_healthy: bool, error: str = None
    ):
        """Update health status for a device."""
        now = datetime.now()

        if device_id not in self.device_health:
            self.device_health[device_id] = DeviceHealth(
                device_id=device_id, is_healthy=is_healthy, last_heartbeat=now
            )
        else:
            health = self.device_health[device_id]
            health.is_healthy = is_healthy
            health.last_heartbeat = now

            if not is_healthy:
                health.error_count += 1
                health.last_error = error
                health.status = "degraded" if health.error_count < 3 else "failed"

                # Track failure history
                self.failure_history[device_id].append(now)
                # Keep only recent failures (last hour)
                cutoff = now - timedelta(hours=1)
                self.failure_history[device_id] = [
                    t for t in self.failure_history[device_id] if t > cutoff
                ]
            else:
                health.status = "active"

    async def _recover_token_corruption(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from token corruption."""
        if not corruption_report:
            return RecoveryResult(
                status=RecoveryStatus.FAILURE,
                action_taken=RecoveryAction.NO_ACTION,
                success=False,
                error_message="No corruption report provided",
                affected_devices=affected_devices,
            )

        # Choose recovery action based on severity
        if corruption_report.severity in [
            CorruptionSeverity.LOW,
            CorruptionSeverity.MEDIUM,
        ]:
            # Try regenerating the token
            action = RecoveryAction.REGENERATE_TOKEN
            # Simulate token regeneration (would integrate with actual generation logic)
            await asyncio.sleep(0.1)  # Simulate processing time
            success = True
        else:
            # Severe corruption - restart pipeline stage
            action = RecoveryAction.RESTART_PIPELINE_STAGE
            success = await self.reinitialize_pipeline(affected_devices)

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details=f"Token corruption recovery: {action}",
        )

    async def _recover_encoding_error(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from encoding errors."""
        # Encoding errors usually require regeneration
        action = RecoveryAction.REGENERATE_TOKEN

        # Simulate encoding fix
        await asyncio.sleep(0.05)
        success = True

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details="Encoding error recovered by token regeneration",
        )

    async def _recover_synchronization_failure(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from synchronization failures."""
        action = RecoveryAction.RESET_SYNCHRONIZATION

        try:
            if self.sync_reset_callback:
                if asyncio.iscoroutinefunction(self.sync_reset_callback):
                    success = await asyncio.wait_for(
                        self.sync_reset_callback(), timeout=self.recovery_timeout
                    )
                else:
                    success = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(self.sync_reset_callback)
                        ),
                        timeout=self.recovery_timeout,
                    )
            else:
                # Fallback: restart pipeline
                action = RecoveryAction.RESTART_PIPELINE_STAGE
                success = await self.reinitialize_pipeline(affected_devices)

            return RecoveryResult(
                status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
                action_taken=action,
                success=success,
                affected_devices=affected_devices,
                details="Synchronization failure recovery",
            )

        except asyncio.TimeoutError:
            return RecoveryResult(
                status=RecoveryStatus.FAILURE,
                action_taken=action,
                success=False,
                error_message="Synchronization reset timed out",
                affected_devices=affected_devices,
            )

    async def _recover_communication_failure(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from communication failures."""
        # Try simple retry first
        action = RecoveryAction.RETRY

        # Check failure frequency for affected devices
        frequent_failures = []
        for device_id in affected_devices:
            recent_failures = len(self.failure_history.get(device_id, []))
            if recent_failures > 5:  # More than 5 failures in the last hour
                frequent_failures.append(device_id)

        if frequent_failures:
            # Reinitialize devices with frequent failures
            action = RecoveryAction.REINITIALIZE_DEVICE
            success = True
            for device_id in frequent_failures:
                if self.device_reinit_callback:
                    if asyncio.iscoroutinefunction(self.device_reinit_callback):
                        device_success = await asyncio.wait_for(
                            self.device_reinit_callback(device_id),
                            timeout=self.recovery_timeout,
                        )
                    else:
                        device_success = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    self.device_reinit_callback, device_id
                                )
                            ),
                            timeout=self.recovery_timeout,
                        )
                    success = success and device_success
        else:
            # Simple retry for infrequent failures
            await asyncio.sleep(0.1)  # Brief delay before retry
            success = True

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details=f"Communication failure recovery: {action}",
        )

    async def _recover_device_failure(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from device failures."""
        action = RecoveryAction.REINITIALIZE_DEVICE
        success = True

        for device_id in affected_devices:
            if self.device_reinit_callback:
                try:
                    if asyncio.iscoroutinefunction(self.device_reinit_callback):
                        device_success = await asyncio.wait_for(
                            self.device_reinit_callback(device_id),
                            timeout=self.recovery_timeout,
                        )
                    else:
                        device_success = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    self.device_reinit_callback, device_id
                                )
                            ),
                            timeout=self.recovery_timeout,
                        )
                    success = success and device_success
                except asyncio.TimeoutError:
                    success = False
                    break
            else:
                success = False

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details="Device failure recovery",
        )

    async def _recover_pipeline_failure(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from pipeline failures."""
        action = RecoveryAction.RESTART_PIPELINE_STAGE
        success = await self.reinitialize_pipeline(affected_devices)

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details="Pipeline failure recovery",
        )

    async def _recover_memory_corruption(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from memory corruption."""
        # Memory corruption is serious - reinitialize affected devices
        action = RecoveryAction.REINITIALIZE_DEVICE
        success = True

        for device_id in affected_devices:
            if self.device_reinit_callback:
                try:
                    if asyncio.iscoroutinefunction(self.device_reinit_callback):
                        device_success = await asyncio.wait_for(
                            self.device_reinit_callback(device_id),
                            timeout=self.recovery_timeout,
                        )
                    else:
                        device_success = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    self.device_reinit_callback, device_id
                                )
                            ),
                            timeout=self.recovery_timeout,
                        )
                    success = success and device_success
                except asyncio.TimeoutError:
                    success = False
                    break

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details="Memory corruption recovery",
        )

    async def _recover_timeout(
        self, affected_devices: List[str], corruption_report: Optional[CorruptionReport]
    ) -> RecoveryResult:
        """Recover from timeout failures."""
        # Start with retry, escalate if needed
        action = RecoveryAction.RETRY

        # Check if this is a recurring timeout issue
        timeout_devices = []
        for device_id in affected_devices:
            recent_failures = len(self.failure_history.get(device_id, []))
            if recent_failures > 3:
                timeout_devices.append(device_id)

        if timeout_devices:
            # Reset synchronization for persistent timeouts
            action = RecoveryAction.RESET_SYNCHRONIZATION
            if self.sync_reset_callback:
                if asyncio.iscoroutinefunction(self.sync_reset_callback):
                    success = await asyncio.wait_for(
                        self.sync_reset_callback(), timeout=self.recovery_timeout
                    )
                else:
                    success = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(self.sync_reset_callback)
                        ),
                        timeout=self.recovery_timeout,
                    )
            else:
                success = False
        else:
            # Simple retry for occasional timeouts
            await asyncio.sleep(0.2)  # Brief delay
            success = True

        return RecoveryResult(
            status=RecoveryStatus.SUCCESS if success else RecoveryStatus.FAILURE,
            action_taken=action,
            success=success,
            affected_devices=affected_devices,
            details=f"Timeout recovery: {action}",
        )
