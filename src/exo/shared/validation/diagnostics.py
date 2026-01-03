"""
Comprehensive logging and diagnostics for distributed inference validation.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from loguru import logger

from ..types.validation import (
    ValidationStatus, CorruptionType, FailureMode, RecoveryAction,
    EnhancedTokenChunk, ValidationResult, RecoveryResult
)
from ..types.models import ModelId


class DiagnosticLevel(str, Enum):
    """Diagnostic logging levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DiagnosticCategory(str, Enum):
    """Categories for diagnostic information."""
    VALIDATION = "validation"
    GENERATION = "generation"
    SYNCHRONIZATION = "synchronization"
    COMMUNICATION = "communication"
    RECOVERY = "recovery"
    PERFORMANCE = "performance"
    CORRUPTION = "corruption"
    DEVICE_HEALTH = "device_health"


@dataclass
class DiagnosticEvent:
    """Structured diagnostic event."""
    timestamp: datetime
    level: DiagnosticLevel
    category: DiagnosticCategory
    event_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    device_id: Optional[str] = None
    model_id: Optional[ModelId] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class PerformanceMetric:
    """Performance metric for diagnostics."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "performance"
    device_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "device_id": self.device_id
        }


@dataclass
class ValidationDiagnostic:
    """Diagnostic information for validation operations."""
    validation_id: str
    token_count: int
    validation_duration_ms: float
    corruption_detected: bool
    corruption_types: List[CorruptionType]
    validation_overhead_ms: float
    device_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validation_id": self.validation_id,
            "token_count": self.token_count,
            "validation_duration_ms": self.validation_duration_ms,
            "corruption_detected": self.corruption_detected,
            "corruption_types": [ct.value for ct in self.corruption_types],
            "validation_overhead_ms": self.validation_overhead_ms,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat()
        }


class DiagnosticLogger:
    """
    Enhanced diagnostic logger for distributed inference validation.
    
    Provides structured logging with performance metrics, corruption tracking,
    and comprehensive debugging information.
    """
    
    def __init__(self, 
                 log_file: Optional[Path] = None,
                 enable_performance_tracking: bool = True,
                 max_events_in_memory: int = 10000):
        """
        Initialize diagnostic logger.
        
        Args:
            log_file: Optional file path for persistent logging
            enable_performance_tracking: Whether to track performance metrics
            max_events_in_memory: Maximum events to keep in memory
        """
        self.log_file = log_file
        self.enable_performance_tracking = enable_performance_tracking
        self.max_events_in_memory = max_events_in_memory
        
        # In-memory storage
        self.events: List[DiagnosticEvent] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.validation_diagnostics: List[ValidationDiagnostic] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self._operation_start_times: Dict[str, float] = {}
        
        # Setup file logging if specified
        if self.log_file:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file-based logging."""
        logger.add(
            self.log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="1 week",
            compression="gz",
            enqueue=True
        )
    
    def log_event(self, 
                  level: DiagnosticLevel,
                  category: DiagnosticCategory,
                  event_type: str,
                  message: str,
                  details: Optional[Dict[str, Any]] = None,
                  device_id: Optional[str] = None,
                  model_id: Optional[ModelId] = None,
                  correlation_id: Optional[str] = None):
        """
        Log a diagnostic event.
        
        Args:
            level: Diagnostic level
            category: Event category
            event_type: Specific event type
            message: Human-readable message
            details: Additional structured details
            device_id: Optional device identifier
            model_id: Optional model identifier
            correlation_id: Optional correlation ID for tracking related events
        """
        event = DiagnosticEvent(
            timestamp=datetime.now(),
            level=level,
            category=category,
            event_type=event_type,
            message=message,
            details=details or {},
            device_id=device_id,
            model_id=model_id,
            correlation_id=correlation_id
        )
        
        with self._lock:
            self.events.append(event)
            
            # Trim events if needed
            if len(self.events) > self.max_events_in_memory:
                self.events = self.events[-self.max_events_in_memory:]
        
        # Log to loguru as well
        log_message = f"[{category.value}:{event_type}] {message}"
        if details:
            log_message += f" | Details: {json.dumps(details, default=str)}"
        
        if level == DiagnosticLevel.TRACE:
            logger.trace(log_message)
        elif level == DiagnosticLevel.DEBUG:
            logger.debug(log_message)
        elif level == DiagnosticLevel.INFO:
            logger.info(log_message)
        elif level == DiagnosticLevel.WARNING:
            logger.warning(log_message)
        elif level == DiagnosticLevel.ERROR:
            logger.error(log_message)
        elif level == DiagnosticLevel.CRITICAL:
            logger.critical(log_message)
    
    def log_validation_start(self, 
                           validation_id: str,
                           token_count: int,
                           device_id: str,
                           correlation_id: Optional[str] = None):
        """Log start of validation operation."""
        self.start_performance_tracking(f"validation_{validation_id}")
        
        self.log_event(
            level=DiagnosticLevel.DEBUG,
            category=DiagnosticCategory.VALIDATION,
            event_type="validation_start",
            message=f"Starting validation for {token_count} tokens",
            details={
                "validation_id": validation_id,
                "token_count": token_count,
                "device_id": device_id
            },
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_validation_complete(self,
                              validation_id: str,
                              token_count: int,
                              corruption_detected: bool,
                              corruption_types: List[CorruptionType],
                              device_id: str,
                              correlation_id: Optional[str] = None):
        """Log completion of validation operation."""
        duration_ms = self.end_performance_tracking(f"validation_{validation_id}")
        
        # Calculate validation overhead (simplified)
        validation_overhead_ms = max(0.0, duration_ms - (token_count * 0.1))  # Assume 0.1ms per token baseline
        
        # Create validation diagnostic
        diagnostic = ValidationDiagnostic(
            validation_id=validation_id,
            token_count=token_count,
            validation_duration_ms=duration_ms,
            corruption_detected=corruption_detected,
            corruption_types=corruption_types,
            validation_overhead_ms=validation_overhead_ms,
            device_id=device_id,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self.validation_diagnostics.append(diagnostic)
        
        self.log_event(
            level=DiagnosticLevel.INFO if not corruption_detected else DiagnosticLevel.WARNING,
            category=DiagnosticCategory.VALIDATION,
            event_type="validation_complete",
            message=f"Validation completed: {token_count} tokens, corruption={'detected' if corruption_detected else 'none'}",
            details={
                "validation_id": validation_id,
                "duration_ms": duration_ms,
                "overhead_ms": validation_overhead_ms,
                "corruption_types": [ct.value for ct in corruption_types]
            },
            device_id=device_id,
            correlation_id=correlation_id
        )
        
        # Log performance metric
        if self.enable_performance_tracking:
            self.log_performance_metric(
                name="validation_duration",
                value=duration_ms,
                unit="ms",
                device_id=device_id
            )
    
    def log_corruption_detected(self,
                              corruption_type: CorruptionType,
                              affected_tokens: List[EnhancedTokenChunk],
                              severity: str,
                              device_id: str,
                              correlation_id: Optional[str] = None):
        """Log corruption detection."""
        self.log_event(
            level=DiagnosticLevel.ERROR,
            category=DiagnosticCategory.CORRUPTION,
            event_type="corruption_detected",
            message=f"Corruption detected: {corruption_type.value} affecting {len(affected_tokens)} tokens",
            details={
                "corruption_type": corruption_type.value,
                "affected_token_count": len(affected_tokens),
                "severity": severity,
                "token_positions": [token.sequence_position for token in affected_tokens[:10]],  # First 10
                "sample_corrupted_text": affected_tokens[0].text[:100] if affected_tokens else ""
            },
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_recovery_attempt(self,
                           failure_mode: FailureMode,
                           recovery_action: RecoveryAction,
                           device_id: str,
                           correlation_id: Optional[str] = None):
        """Log recovery attempt."""
        recovery_id = f"recovery_{int(time.time() * 1000)}"
        self.start_performance_tracking(recovery_id)
        
        self.log_event(
            level=DiagnosticLevel.WARNING,
            category=DiagnosticCategory.RECOVERY,
            event_type="recovery_attempt",
            message=f"Attempting recovery: {recovery_action.value} for {failure_mode.value}",
            details={
                "recovery_id": recovery_id,
                "failure_mode": failure_mode.value,
                "recovery_action": recovery_action.value
            },
            device_id=device_id,
            correlation_id=correlation_id
        )
        
        return recovery_id
    
    def log_recovery_result(self,
                          recovery_id: str,
                          result: RecoveryResult,
                          device_id: str,
                          correlation_id: Optional[str] = None):
        """Log recovery result."""
        duration_ms = self.end_performance_tracking(recovery_id)
        
        self.log_event(
            level=DiagnosticLevel.INFO if result.success else DiagnosticLevel.ERROR,
            category=DiagnosticCategory.RECOVERY,
            event_type="recovery_complete",
            message=f"Recovery {'successful' if result.success else 'failed'}: {result.action_taken.value}",
            details={
                "recovery_id": recovery_id,
                "success": result.success,
                "duration_ms": duration_ms,
                "attempts": result.attempts,
                "error_message": result.error_message,
                "affected_devices": result.affected_devices
            },
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_generation_pipeline_event(self,
                                    event_type: str,
                                    message: str,
                                    token_info: Optional[Dict[str, Any]] = None,
                                    device_id: Optional[str] = None,
                                    correlation_id: Optional[str] = None):
        """Log generation pipeline events."""
        self.log_event(
            level=DiagnosticLevel.DEBUG,
            category=DiagnosticCategory.GENERATION,
            event_type=event_type,
            message=message,
            details=token_info or {},
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_synchronization_event(self,
                                event_type: str,
                                message: str,
                                sync_details: Optional[Dict[str, Any]] = None,
                                device_id: Optional[str] = None,
                                correlation_id: Optional[str] = None):
        """Log synchronization events."""
        level = DiagnosticLevel.WARNING if "timeout" in event_type or "failure" in event_type else DiagnosticLevel.DEBUG
        
        self.log_event(
            level=level,
            category=DiagnosticCategory.SYNCHRONIZATION,
            event_type=event_type,
            message=message,
            details=sync_details or {},
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_communication_event(self,
                              event_type: str,
                              message: str,
                              comm_details: Optional[Dict[str, Any]] = None,
                              device_id: Optional[str] = None,
                              correlation_id: Optional[str] = None):
        """Log communication events."""
        level = DiagnosticLevel.WARNING if "error" in event_type or "failure" in event_type else DiagnosticLevel.DEBUG
        
        self.log_event(
            level=level,
            category=DiagnosticCategory.COMMUNICATION,
            event_type=event_type,
            message=message,
            details=comm_details or {},
            device_id=device_id,
            correlation_id=correlation_id
        )
    
    def log_performance_metric(self,
                             name: str,
                             value: float,
                             unit: str,
                             device_id: Optional[str] = None,
                             category: str = "performance"):
        """Log a performance metric."""
        if not self.enable_performance_tracking:
            return
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            device_id=device_id
        )
        
        with self._lock:
            self.performance_metrics.append(metric)
            
            # Trim metrics if needed
            if len(self.performance_metrics) > self.max_events_in_memory:
                self.performance_metrics = self.performance_metrics[-self.max_events_in_memory:]
        
        self.log_event(
            level=DiagnosticLevel.TRACE,
            category=DiagnosticCategory.PERFORMANCE,
            event_type="metric_recorded",
            message=f"Performance metric: {name} = {value} {unit}",
            details={
                "metric_name": name,
                "value": value,
                "unit": unit,
                "category": category
            },
            device_id=device_id
        )
    
    def start_performance_tracking(self, operation_id: str) -> str:
        """Start tracking performance for an operation."""
        if not self.enable_performance_tracking:
            return operation_id
        
        with self._lock:
            self._operation_start_times[operation_id] = time.time()
        
        return operation_id
    
    def end_performance_tracking(self, operation_id: str) -> float:
        """End performance tracking and return duration in milliseconds."""
        if not self.enable_performance_tracking:
            return 0.0
        
        with self._lock:
            start_time = self._operation_start_times.pop(operation_id, None)
        
        if start_time is None:
            return 0.0
        
        duration_ms = (time.time() - start_time) * 1000
        return duration_ms
    
    def get_diagnostic_summary(self, 
                             time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get diagnostic summary for analysis."""
        with self._lock:
            events = self.events.copy()
            metrics = self.performance_metrics.copy()
            validations = self.validation_diagnostics.copy()
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            events = [e for e in events if e.timestamp >= cutoff_time]
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            validations = [v for v in validations if v.timestamp >= cutoff_time]
        
        # Event statistics
        event_counts = {}
        error_counts = {}
        
        for event in events:
            category = event.category.value
            event_counts[category] = event_counts.get(category, 0) + 1
            
            if event.level in [DiagnosticLevel.ERROR, DiagnosticLevel.CRITICAL]:
                error_counts[category] = error_counts.get(category, 0) + 1
        
        # Performance statistics
        perf_stats = {}
        if metrics:
            metric_groups = {}
            for metric in metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            for name, values in metric_groups.items():
                perf_stats[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Validation statistics
        validation_stats = {}
        if validations:
            total_validations = len(validations)
            corrupted_validations = sum(1 for v in validations if v.corruption_detected)
            avg_duration = sum(v.validation_duration_ms for v in validations) / total_validations
            avg_overhead = sum(v.validation_overhead_ms for v in validations) / total_validations
            
            validation_stats = {
                "total_validations": total_validations,
                "corruption_rate": corrupted_validations / total_validations,
                "average_duration_ms": avg_duration,
                "average_overhead_ms": avg_overhead
            }
        
        return {
            "summary_period": time_window.total_seconds() if time_window else "all_time",
            "total_events": len(events),
            "event_counts_by_category": event_counts,
            "error_counts_by_category": error_counts,
            "performance_statistics": perf_stats,
            "validation_statistics": validation_stats,
            "last_event_time": events[-1].timestamp.isoformat() if events else None
        }
    
    def export_diagnostics(self, 
                         output_file: Path,
                         time_window: Optional[timedelta] = None,
                         include_raw_events: bool = True) -> bool:
        """Export diagnostic data to file."""
        try:
            with self._lock:
                events = self.events.copy()
                metrics = self.performance_metrics.copy()
                validations = self.validation_diagnostics.copy()
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                events = [e for e in events if e.timestamp >= cutoff_time]
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                validations = [v for v in validations if v.timestamp >= cutoff_time]
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_window_seconds": time_window.total_seconds() if time_window else None,
                "summary": self.get_diagnostic_summary(time_window),
                "validation_diagnostics": [v.to_dict() for v in validations],
                "performance_metrics": [m.to_dict() for m in metrics]
            }
            
            if include_raw_events:
                export_data["events"] = [e.to_dict() for e in events]
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export diagnostics: {e}")
            return False


# Global diagnostic logger instance
_diagnostic_logger: Optional[DiagnosticLogger] = None


def get_diagnostic_logger() -> DiagnosticLogger:
    """Get the global diagnostic logger instance."""
    global _diagnostic_logger
    if _diagnostic_logger is None:
        _diagnostic_logger = DiagnosticLogger()
    return _diagnostic_logger


def setup_diagnostics(log_file: Optional[Path] = None,
                     enable_performance_tracking: bool = True,
                     max_events_in_memory: int = 10000) -> DiagnosticLogger:
    """Setup global diagnostic logging."""
    global _diagnostic_logger
    _diagnostic_logger = DiagnosticLogger(
        log_file=log_file,
        enable_performance_tracking=enable_performance_tracking,
        max_events_in_memory=max_events_in_memory
    )
    return _diagnostic_logger


# Convenience functions for common diagnostic operations
def log_validation_event(event_type: str, message: str, **kwargs):
    """Log a validation event."""
    get_diagnostic_logger().log_event(
        level=DiagnosticLevel.DEBUG,
        category=DiagnosticCategory.VALIDATION,
        event_type=event_type,
        message=message,
        **kwargs
    )


def log_corruption_event(corruption_type: CorruptionType, message: str, **kwargs):
    """Log a corruption event."""
    get_diagnostic_logger().log_event(
        level=DiagnosticLevel.ERROR,
        category=DiagnosticCategory.CORRUPTION,
        event_type="corruption_detected",
        message=message,
        details={"corruption_type": corruption_type.value},
        **kwargs
    )


def log_performance_event(metric_name: str, value: float, unit: str, **kwargs):
    """Log a performance metric."""
    get_diagnostic_logger().log_performance_metric(
        name=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )