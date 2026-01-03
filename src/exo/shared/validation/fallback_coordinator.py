"""
FallbackCoordinator for graceful degradation to single-device mode.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set, Any
from enum import Enum

from ..types.validation import (
    FailureMode, RecoveryResult, DeviceHealth, CorruptionReport,
    CorruptionSeverity, RecoveryStatus
)


logger = logging.getLogger(__name__)


class FallbackTrigger(str, Enum):
    """Conditions that can trigger fallback to single-device mode."""
    PERSISTENT_CORRUPTION = "persistent_corruption"
    MULTIPLE_DEVICE_FAILURES = "multiple_device_failures"
    COMMUNICATION_BREAKDOWN = "communication_breakdown"
    RECOVERY_FAILURE = "recovery_failure"
    MANUAL_TRIGGER = "manual_trigger"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class FallbackStatus(str, Enum):
    """Status of fallback operations."""
    DISTRIBUTED = "distributed"
    FALLBACK_INITIATED = "fallback_initiated"
    FALLBACK_IN_PROGRESS = "fallback_in_progress"
    SINGLE_DEVICE = "single_device"
    FALLBACK_FAILED = "fallback_failed"
    RECOVERY_ATTEMPTED = "recovery_attempted"


class FallbackDecision:
    """Decision about whether to trigger fallback."""
    
    def __init__(self, should_fallback: bool, trigger: Optional[FallbackTrigger] = None,
                 reason: str = "", confidence: float = 0.0):
        self.should_fallback = should_fallback
        self.trigger = trigger
        self.reason = reason
        self.confidence = confidence  # 0.0 to 1.0
        self.timestamp = datetime.now()


class FallbackCoordinator:
    """
    Coordinates graceful degradation from distributed to single-device inference
    when distributed mode becomes unreliable or fails persistently.
    """
    
    def __init__(self, 
                 corruption_threshold: int = 5,
                 device_failure_threshold: float = 0.5,
                 recovery_failure_threshold: int = 3,
                 fallback_timeout: float = 60.0):
        """
        Initialize FallbackCoordinator.
        
        Args:
            corruption_threshold: Number of corruption events before considering fallback
            device_failure_threshold: Fraction of devices that must fail to trigger fallback
            recovery_failure_threshold: Number of failed recovery attempts before fallback
            fallback_timeout: Maximum time to wait for fallback completion
        """
        self.corruption_threshold = corruption_threshold
        self.device_failure_threshold = device_failure_threshold
        self.recovery_failure_threshold = recovery_failure_threshold
        self.fallback_timeout = fallback_timeout
        
        # Current state
        self.status = FallbackStatus.DISTRIBUTED
        self.active_devices: Set[str] = set()
        self.failed_devices: Set[str] = set()
        self.primary_device: Optional[str] = None
        
        # Event tracking
        self.corruption_events: List[datetime] = []
        self.recovery_failures: List[datetime] = []
        self.device_health_history: Dict[str, List[DeviceHealth]] = {}
        
        # Callbacks for coordination
        self.master_notification_callback: Optional[Callable[[FallbackStatus], None]] = None
        self.worker_shutdown_callback: Optional[Callable[[List[str]], bool]] = None
        self.single_device_init_callback: Optional[Callable[[str], bool]] = None
        self.distributed_restore_callback: Optional[Callable[[List[str]], bool]] = None
        
        # Statistics
        self.fallback_history: List[Dict[str, Any]] = []
        self.total_fallbacks = 0
        self.successful_fallbacks = 0
        
    def register_callbacks(self,
                          master_notification: Optional[Callable[[FallbackStatus], None]] = None,
                          worker_shutdown: Optional[Callable[[List[str]], bool]] = None,
                          single_device_init: Optional[Callable[[str], bool]] = None,
                          distributed_restore: Optional[Callable[[List[str]], bool]] = None):
        """Register callbacks for coordinating fallback operations."""
        if master_notification:
            self.master_notification_callback = master_notification
        if worker_shutdown:
            self.worker_shutdown_callback = worker_shutdown
        if single_device_init:
            self.single_device_init_callback = single_device_init
        if distributed_restore:
            self.distributed_restore_callback = distributed_restore
    
    def update_device_status(self, device_id: str, health: DeviceHealth):
        """Update the health status of a device."""
        if device_id not in self.device_health_history:
            self.device_health_history[device_id] = []
        
        self.device_health_history[device_id].append(health)
        
        # Keep only recent history (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.device_health_history[device_id] = [
            h for h in self.device_health_history[device_id] 
            if h.last_heartbeat > cutoff
        ]
        
        # Update active/failed device sets
        if health.is_healthy:
            self.active_devices.add(device_id)
            self.failed_devices.discard(device_id)
        else:
            self.failed_devices.add(device_id)
            self.active_devices.discard(device_id)
    
    def report_corruption_event(self, corruption_report: CorruptionReport):
        """Report a corruption event for fallback decision making."""
        now = datetime.now()
        self.corruption_events.append(now)
        
        # Keep only recent events (last 10 minutes)
        cutoff = now - timedelta(minutes=10)
        self.corruption_events = [t for t in self.corruption_events if t > cutoff]
        
        logger.info(f"Corruption event reported: {corruption_report.corruption_type}, "
                   f"total recent events: {len(self.corruption_events)}")
    
    def report_recovery_failure(self, recovery_result: RecoveryResult):
        """Report a failed recovery attempt."""
        if not recovery_result.success:
            now = datetime.now()
            self.recovery_failures.append(now)
            
            # Keep only recent failures (last 10 minutes)
            cutoff = now - timedelta(minutes=10)
            self.recovery_failures = [t for t in self.recovery_failures if t > cutoff]
            
            logger.warning(f"Recovery failure reported: {recovery_result.action_taken}, "
                          f"total recent failures: {len(self.recovery_failures)}")
    
    def should_trigger_fallback(self) -> FallbackDecision:
        """
        Determine if fallback to single-device mode should be triggered.
        
        Returns:
            FallbackDecision with recommendation and reasoning
        """
        if self.status != FallbackStatus.DISTRIBUTED:
            return FallbackDecision(False, reason="Already in fallback or transitioning")
        
        # Check corruption threshold
        if len(self.corruption_events) >= self.corruption_threshold:
            return FallbackDecision(
                True, 
                FallbackTrigger.PERSISTENT_CORRUPTION,
                f"Corruption events ({len(self.corruption_events)}) exceed threshold ({self.corruption_threshold})",
                confidence=0.9
            )
        
        # Check device failure threshold
        total_devices = len(self.active_devices) + len(self.failed_devices)
        if total_devices > 0:
            failure_rate = len(self.failed_devices) / total_devices
            if failure_rate >= self.device_failure_threshold:
                return FallbackDecision(
                    True,
                    FallbackTrigger.MULTIPLE_DEVICE_FAILURES,
                    f"Device failure rate ({failure_rate:.2f}) exceeds threshold ({self.device_failure_threshold})",
                    confidence=0.8
                )
        
        # Check recovery failure threshold
        if len(self.recovery_failures) >= self.recovery_failure_threshold:
            return FallbackDecision(
                True,
                FallbackTrigger.RECOVERY_FAILURE,
                f"Recovery failures ({len(self.recovery_failures)}) exceed threshold ({self.recovery_failure_threshold})",
                confidence=0.85
            )
        
        # Check for communication breakdown (no active devices)
        if len(self.active_devices) == 0 and len(self.failed_devices) > 0:
            return FallbackDecision(
                True,
                FallbackTrigger.COMMUNICATION_BREAKDOWN,
                "No active devices remaining in distributed system",
                confidence=1.0
            )
        
        return FallbackDecision(False, reason="No fallback conditions met")
    
    async def initiate_fallback(self, trigger: Optional[FallbackTrigger] = None) -> bool:
        """
        Initiate fallback to single-device mode.
        
        Args:
            trigger: Optional specific trigger for the fallback
            
        Returns:
            True if fallback was successful, False otherwise
        """
        if self.status != FallbackStatus.DISTRIBUTED:
            logger.warning(f"Cannot initiate fallback from status: {self.status}")
            return False
        
        logger.info(f"Initiating fallback to single-device mode, trigger: {trigger}")
        
        self.status = FallbackStatus.FALLBACK_INITIATED
        self.total_fallbacks += 1
        
        # Notify master about fallback initiation
        if self.master_notification_callback:
            try:
                self.master_notification_callback(self.status)
            except Exception as e:
                logger.error(f"Failed to notify master about fallback: {e}")
        
        try:
            # Step 1: Select primary device for single-device mode
            primary_device = self._select_primary_device()
            if not primary_device:
                logger.error("No suitable device found for single-device mode")
                self.status = FallbackStatus.FALLBACK_FAILED
                return False
            
            self.primary_device = primary_device
            self.status = FallbackStatus.FALLBACK_IN_PROGRESS
            
            # Step 2: Shutdown other workers
            other_devices = [d for d in self.active_devices if d != primary_device]
            if other_devices and self.worker_shutdown_callback:
                if asyncio.iscoroutinefunction(self.worker_shutdown_callback):
                    shutdown_success = await asyncio.wait_for(
                        self.worker_shutdown_callback(other_devices),
                        timeout=self.fallback_timeout / 2
                    )
                else:
                    shutdown_success = await asyncio.wait_for(
                        asyncio.create_task(asyncio.to_thread(self.worker_shutdown_callback, other_devices)),
                        timeout=self.fallback_timeout / 2
                    )
                
                if not shutdown_success:
                    logger.warning("Some workers failed to shutdown gracefully")
            
            # Step 3: Initialize single-device mode on primary device
            if self.single_device_init_callback:
                if asyncio.iscoroutinefunction(self.single_device_init_callback):
                    init_success = await asyncio.wait_for(
                        self.single_device_init_callback(primary_device),
                        timeout=self.fallback_timeout / 2
                    )
                else:
                    init_success = await asyncio.wait_for(
                        asyncio.create_task(asyncio.to_thread(self.single_device_init_callback, primary_device)),
                        timeout=self.fallback_timeout / 2
                    )
                
                if not init_success:
                    logger.error("Failed to initialize single-device mode")
                    self.status = FallbackStatus.FALLBACK_FAILED
                    return False
            
            # Step 4: Update status and record success
            self.status = FallbackStatus.SINGLE_DEVICE
            self.successful_fallbacks += 1
            
            # Record fallback event
            self.fallback_history.append({
                'timestamp': datetime.now(),
                'trigger': trigger,
                'primary_device': primary_device,
                'failed_devices': list(self.failed_devices),
                'success': True
            })
            
            logger.info(f"Fallback completed successfully, primary device: {primary_device}")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Fallback timed out after {self.fallback_timeout}s")
            self.status = FallbackStatus.FALLBACK_FAILED
            return False
        except Exception as e:
            logger.error(f"Fallback failed with exception: {e}")
            self.status = FallbackStatus.FALLBACK_FAILED
            return False
    
    async def attempt_recovery_to_distributed(self) -> bool:
        """
        Attempt to recover from single-device mode back to distributed mode.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        if self.status != FallbackStatus.SINGLE_DEVICE:
            logger.warning(f"Cannot recover to distributed from status: {self.status}")
            return False
        
        logger.info("Attempting recovery to distributed mode")
        
        self.status = FallbackStatus.RECOVERY_ATTEMPTED
        
        try:
            # Check if enough devices are healthy again
            healthy_devices = [
                device_id for device_id, health_list in self.device_health_history.items()
                if health_list and health_list[-1].is_healthy
            ]
            
            if len(healthy_devices) < 2:
                logger.info("Insufficient healthy devices for distributed mode")
                self.status = FallbackStatus.SINGLE_DEVICE
                return False
            
            # Attempt to restore distributed mode
            if self.distributed_restore_callback:
                if asyncio.iscoroutinefunction(self.distributed_restore_callback):
                    restore_success = await asyncio.wait_for(
                        self.distributed_restore_callback(healthy_devices),
                        timeout=self.fallback_timeout
                    )
                else:
                    restore_success = await asyncio.wait_for(
                        asyncio.create_task(asyncio.to_thread(self.distributed_restore_callback, healthy_devices)),
                        timeout=self.fallback_timeout
                    )
                
                if restore_success:
                    self.status = FallbackStatus.DISTRIBUTED
                    self.active_devices = set(healthy_devices)
                    self.failed_devices.clear()
                    self.primary_device = None
                    
                    # Clear recent failure history
                    self.corruption_events.clear()
                    self.recovery_failures.clear()
                    
                    logger.info("Successfully recovered to distributed mode")
                    return True
                else:
                    logger.warning("Failed to restore distributed mode")
                    self.status = FallbackStatus.SINGLE_DEVICE
                    return False
            else:
                logger.warning("No distributed restore callback registered")
                self.status = FallbackStatus.SINGLE_DEVICE
                return False
                
        except asyncio.TimeoutError:
            logger.error("Recovery to distributed mode timed out")
            self.status = FallbackStatus.SINGLE_DEVICE
            return False
        except Exception as e:
            logger.error(f"Recovery to distributed mode failed: {e}")
            self.status = FallbackStatus.SINGLE_DEVICE
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current fallback coordinator status."""
        return {
            'status': self.status,
            'active_devices': list(self.active_devices),
            'failed_devices': list(self.failed_devices),
            'primary_device': self.primary_device,
            'recent_corruption_events': len(self.corruption_events),
            'recent_recovery_failures': len(self.recovery_failures),
            'total_fallbacks': self.total_fallbacks,
            'successful_fallbacks': self.successful_fallbacks,
            'fallback_success_rate': (
                self.successful_fallbacks / self.total_fallbacks 
                if self.total_fallbacks > 0 else 0.0
            )
        }
    
    def _select_primary_device(self) -> Optional[str]:
        """Select the best device to use for single-device mode."""
        if not self.active_devices:
            return None
        
        # Score devices based on health history
        device_scores = {}
        
        for device_id in self.active_devices:
            health_history = self.device_health_history.get(device_id, [])
            if not health_history:
                device_scores[device_id] = 0.0
                continue
            
            # Calculate score based on recent health
            recent_health = health_history[-10:]  # Last 10 health reports
            healthy_count = sum(1 for h in recent_health if h.is_healthy)
            health_ratio = healthy_count / len(recent_health)
            
            # Factor in error count (lower is better)
            latest_health = health_history[-1]
            error_penalty = min(latest_health.error_count * 0.1, 0.5)
            
            device_scores[device_id] = health_ratio - error_penalty
        
        # Select device with highest score
        if device_scores:
            best_device = max(device_scores.items(), key=lambda x: x[1])
            logger.info(f"Selected primary device: {best_device[0]} (score: {best_device[1]:.2f})")
            return best_device[0]
        
        # Fallback to first available device
        return next(iter(self.active_devices))